"""
core/window.py

Builds the six input tensors the ONNX graph expects, for one station and one
forecast origin.

The contract, all of it verified empirically rather than assumed:

  encoder_cont [1, 168, 15]  float32   x_reals order
  decoder_cont [1,  24, 15]  float32   same order
  encoder_cat  [1, 168,  9]  int64     x_categoricals order
  decoder_cat  [1,  24,  9]  int64     same order
  encoder_lengths [1]        int64     always 168
  decoder_lengths [1]        int64     always 24

Two things are easy to get wrong and produce plausible, wrong forecasts:

1. Panel values are ALREADY scaled once (the training RobustScaler was applied
   before val_data.csv was written). TimeSeriesDataSet then fitted a SECOND
   scaler on top, and the model consumes that. panel_column_transforms is that
   second layer and it must be applied here.

2. encoder_length and relative_time_idx never appear in the dataframe. Both are
   divided by max_encoder_length: encoder_length is a constant 1.0, and
   relative_time_idx runs -1.0 .. -1/168 across the encoder and 0 .. 23/168
   across the decoder.

Origin convention: the origin is the LAST OBSERVED hour. The encoder covers
[origin-167, origin] and the forecast covers [origin+1, origin+24].
"""

from dataclasses import dataclass

import numpy as np

from core import config, loader


class WindowError(ValueError):
    """Raised when a window cannot be built. Never silently zero-padded."""


@dataclass(frozen=True)
class ModelInputs:
    encoder_cont: np.ndarray
    decoder_cont: np.ndarray
    encoder_cat: np.ndarray
    decoder_cat: np.ndarray
    encoder_lengths: np.ndarray
    decoder_lengths: np.ndarray
    origin_time_index: int
    origin_timestamp: object
    horizon_timestamps: list

    def as_feed(self):
        return {
            "encoder_cont": self.encoder_cont,
            "decoder_cont": self.decoder_cont,
            "encoder_cat": self.encoder_cat,
            "decoder_cat": self.decoder_cat,
            "encoder_lengths": self.encoder_lengths,
            "decoder_lengths": self.decoder_lengths,
        }


def _scale(values, transform):
    center = transform["center"]
    scale = transform["scale"]
    if scale == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - center) / scale).astype(np.float32)


def _encode(series, mapping, column):
    """Map categorical values to indices, falling back to 0 for unseen ones."""
    out = np.empty(len(series), dtype=np.int64)
    unseen = 0
    for position, value in enumerate(series):
        key = str(value)
        index = mapping.get(key)
        if index is None:
            # Vocabulary index 0 is the fallback slot. Reaching this means the
            # panel holds a value the model never trained on.
            index = 0
            unseen += 1
        out[position] = index
    if unseen:
        raise WindowError(
            f"{unseen} value(s) in '{column}' are outside the model vocabulary"
        )
    return out


def build_inputs(station_id, origin_time_index, mask_future_unknowns=False):
    """Assemble model inputs for one station and origin.

    mask_future_unknowns zeroes the decoder channels the model does not read
    (everything outside time_varying_reals_decoder). The output is unchanged
    either way — leaving it False reproduces a training batch exactly, which
    is what the parity check compares against.
    """
    prep = loader.load_preprocessing()
    frame = loader.station_frame(station_id)

    x_reals = prep["x_reals"]
    x_cats = prep["x_categoricals"]
    vocab = prep["vocab"]
    derived = prep["derived_scalers"]
    panel_transforms = prep["panel_column_transforms"]
    decoder_reals = set(prep.get("time_varying_known_reals", ["Fee"]))

    enc_len = config.ENCODER_LENGTH
    horizon = config.HORIZON

    origin = int(origin_time_index)
    encoder_idx = list(range(origin - enc_len + 1, origin + 1))
    decoder_idx = list(range(origin + 1, origin + 1 + horizon))

    available = frame.index
    missing_enc = [i for i in encoder_idx if i not in available]
    missing_dec = [i for i in decoder_idx if i not in available]
    if missing_enc:
        raise WindowError(
            f"{station_id}: {len(missing_enc)} encoder hour(s) missing before "
            f"origin {origin}. Need {enc_len} contiguous hours."
        )
    if missing_dec:
        raise WindowError(
            f"{station_id}: {len(missing_dec)} horizon hour(s) missing after "
            f"origin {origin}. Need {horizon} contiguous hours."
        )

    enc_rows = frame.loc[encoder_idx]
    dec_rows = frame.loc[decoder_idx]

    # ---- continuous ------------------------------------------------------
    encoder_cont = np.zeros((enc_len, len(x_reals)), dtype=np.float32)
    decoder_cont = np.zeros((horizon, len(x_reals)), dtype=np.float32)

    # relative_time_idx: -enc_len .. -1 across the encoder, 0 .. 23 ahead.
    rel_raw_enc = np.arange(-enc_len, 0, dtype=np.float64)
    rel_raw_dec = np.arange(0, horizon, dtype=np.float64)

    for channel, name in enumerate(x_reals):
        if name == "encoder_length":
            transform = derived["encoder_length"]
            encoder_cont[:, channel] = _scale(
                np.full(enc_len, float(enc_len)), transform
            )
            decoder_cont[:, channel] = _scale(
                np.full(horizon, float(enc_len)), transform
            )
            continue

        if name == "relative_time_idx":
            transform = derived["relative_time_idx"]
            encoder_cont[:, channel] = _scale(rel_raw_enc, transform)
            decoder_cont[:, channel] = _scale(rel_raw_dec, transform)
            continue

        transform = panel_transforms.get(name)
        if transform is None:
            raise WindowError(f"No transform recorded for real column '{name}'")

        encoder_cont[:, channel] = _scale(
            enc_rows[name].to_numpy(dtype=np.float64), transform
        )

        if mask_future_unknowns and name not in decoder_reals:
            # The decoder's variable selection ignores these channels. Zeroing
            # them proves no future information leaks into the forecast.
            decoder_cont[:, channel] = 0.0
        else:
            decoder_cont[:, channel] = _scale(
                dec_rows[name].to_numpy(dtype=np.float64), transform
            )

    # ---- categorical -----------------------------------------------------
    encoder_cat = np.zeros((enc_len, len(x_cats)), dtype=np.int64)
    decoder_cat = np.zeros((horizon, len(x_cats)), dtype=np.int64)

    for channel, name in enumerate(x_cats):
        mapping = vocab[name]
        encoder_cat[:, channel] = _encode(enc_rows[name].tolist(), mapping, name)
        decoder_cat[:, channel] = _encode(dec_rows[name].tolist(), mapping, name)

    return ModelInputs(
        encoder_cont=encoder_cont[np.newaxis, ...],
        decoder_cont=decoder_cont[np.newaxis, ...],
        encoder_cat=encoder_cat[np.newaxis, ...],
        decoder_cat=decoder_cat[np.newaxis, ...],
        encoder_lengths=np.array([enc_len], dtype=np.int64),
        decoder_lengths=np.array([horizon], dtype=np.int64),
        origin_time_index=origin,
        origin_timestamp=enc_rows["Timestamp"].iloc[-1],
        horizon_timestamps=dec_rows["Timestamp"].tolist(),
    )


def observed_history(station_id, origin_time_index, hours=48):
    """Actual demand leading up to the origin, for the chart's left half."""
    frame = loader.station_frame(station_id)
    start = int(origin_time_index) - hours + 1
    idx = [i for i in range(start, int(origin_time_index) + 1) if i in frame.index]
    return frame.loc[idx, ["Timestamp", "Energy_kWh"]].reset_index(drop=True)


def ground_truth(station_id, origin_time_index):
    """Actuals over the forecast horizon, where available."""
    frame = loader.station_frame(station_id)
    idx = [
        i
        for i in range(int(origin_time_index) + 1, int(origin_time_index) + 1 + config.HORIZON)
        if i in frame.index
    ]
    if not idx:
        return None
    return frame.loc[idx, ["Timestamp", "Energy_kWh"]].reset_index(drop=True)
