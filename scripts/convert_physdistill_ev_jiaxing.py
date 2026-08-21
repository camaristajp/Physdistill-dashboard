"""
convert_physdistill_ev_jiaxing.py

Exports the Jiaxing PhysDistill-EV student to ONNX for the dashboard backend.

Run from the dashboard root with the venv active:

    python scripts\\convert_physdistill_ev_jiaxing.py

Requires train_all_jiaxingalpha.py in scripts/ and physdistill_ev.ckpt in
artifacts/. Expected result: artifacts/physdistill_ev_jiaxing.onnx with input
signature encoder_cont [batch, 168, 15] and Location_Information (13, 7).
"""

import collections
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "artifacts" / "physdistill_ev.ckpt"
ONNX_PATH = REPO_ROOT / "artifacts" / "physdistill_ev_jiaxing.onnx"
SCRIPTS_DIR = REPO_ROOT / "scripts"

EXPECTED_STATIC_CATS = {"Location_Information", "District_Name"}
FORBIDDEN_CATS = {"Station_Name", "Postal_Code"}

# Fallbacks for hparams that older checkpoints may not carry. Both are stated
# in student_tft_config.json and confirmed by the ONNX output shape at step 7.
DEFAULT_LOOKBACK = 168
DEFAULT_HORIZON = 24


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def hparam(model, key, default=None):
    """hparams contents vary across checkpoint versions; degrade gracefully."""
    try:
        value = model.hparams[key]
    except (KeyError, AttributeError, TypeError):
        value = None
    if value is None and hasattr(model.hparams, "get"):
        value = model.hparams.get(key, None)
    return default if value is None else value


# --------------------------------------------------------------------------
banner("1. Pin torchmetrics to CPU")

# SparseQuantileLoss subclasses a torchmetrics Metric that recorded a CUDA
# device at training time. Lightning's .to("cpu") makes torchmetrics allocate
# a probe tensor on that stored device, which fails on a CPU-only torch build.
# The loss plays no part in inference, so forcing CPU here is safe.
import torchmetrics  # noqa: E402

torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))
print("  torchmetrics.Metric.device -> cpu")


# --------------------------------------------------------------------------
banner("2. Import custom classes from the Jiaxing training script")

TRAIN_FILE = SCRIPTS_DIR / "train_all_jiaxingalpha.py"
if not TRAIN_FILE.exists():
    print(f"  FAILED: {TRAIN_FILE} not found")
    sys.exit(1)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from train_all_jiaxingalpha import (  # noqa: F401
        QUANTILES,
        RawLossTFT,
        SparseQuantileLoss,
        StudentKDTFT,
    )
except ImportError as exc:
    missing = getattr(exc, "name", None) or str(exc).split()[-1].strip("'\"")
    print(f"  FAILED: {exc}")
    if missing and missing != "train_all_jiaxingalpha":
        print(f"\n  The training script imports '{missing}', which is not")
        print("  installed in this environment. Install it and rerun:")
        print(f"\n      pip install {missing}\n")
    sys.exit(1)

print(f"  OK — quantiles: {QUANTILES}")


# --------------------------------------------------------------------------
banner("3. Load checkpoint")

if not CKPT_PATH.exists():
    print(f"  FAILED: {CKPT_PATH} not found")
    sys.exit(1)

try:
    model = StudentKDTFT.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
except AssertionError as exc:
    if "CUDA" in str(exc):
        print(f"  FAILED: {exc}")
        print("\n  A metric still reports a CUDA device. Retry with the")
        print("  blunter fallback: add this line above the import block —")
        print("      torchmetrics.Metric._apply = lambda self, fn, *a, **k: self")
        sys.exit(1)
    raise

model.eval()

LOOKBACK = int(hparam(model, "max_encoder_length", DEFAULT_LOOKBACK))
HORIZON = int(hparam(model, "max_prediction_length", DEFAULT_HORIZON))

n_params = sum(p.numel() for p in model.parameters())
print(f"  parameters      : {n_params:,}")
print(f"  encoder length  : {LOOKBACK}")
print(f"  prediction len  : {HORIZON}")

if hparam(model, "max_prediction_length") is None:
    print(f"  note: horizon absent from hparams, defaulting to {DEFAULT_HORIZON}")


# --------------------------------------------------------------------------
banner("4. Verify this is the Jiaxing model")

emb_sizes = hparam(model, "embedding_sizes", {})
real_names = list(hparam(model, "x_reals", []))
cat_names = list(emb_sizes.keys())

if not emb_sizes or not real_names:
    print("  FAILED: checkpoint carries no embedding_sizes or x_reals")
    sys.exit(1)

found_forbidden = FORBIDDEN_CATS.intersection(cat_names)
if found_forbidden:
    print(f"  FAILED: found Palo Alto covariates {sorted(found_forbidden)}")
    print("  This checkpoint is not the Jiaxing student. Aborting.")
    sys.exit(1)

missing_cats = EXPECTED_STATIC_CATS.difference(cat_names)
if missing_cats:
    print(f"  FAILED: missing expected Jiaxing covariates {sorted(missing_cats)}")
    sys.exit(1)

n_stations = emb_sizes["Location_Information"][0]
print(f"  station embedding cardinality : {n_stations}")
print(f"  categoricals ({len(cat_names)}) : {cat_names}")
print(f"  continuous   ({len(real_names)}) : {real_names}")

if n_stations != 13:
    print(f"  WARNING: expected 13 stations, found {n_stations}")

N_CONT = len(real_names)
N_CAT = len(cat_names)


# --------------------------------------------------------------------------
banner("5. Export to ONNX")


class PhysDistillEVWrapper(torch.nn.Module):
    """Flat tensors in, quantile predictions out. The TFT wants a dict."""

    def __init__(self, tft):
        super().__init__()
        self.model = tft

    def forward(
        self,
        encoder_cont,
        decoder_cont,
        encoder_cat,
        decoder_cat,
        encoder_lengths,
        decoder_lengths,
    ):
        batch = encoder_cont.size(0)
        x = {
            "encoder_cont": encoder_cont,
            "decoder_cont": decoder_cont,
            "encoder_cat": encoder_cat,
            "decoder_cat": decoder_cat,
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
            # GroupNormalizer(transformation=None, center=False,
            # scale_by_group=False) is an identity transform -> [0, 1] per row
            "target_scale": torch.zeros(batch, 2)
            .float()
            .index_fill_(1, torch.tensor([1]), 1.0),
            "encoder_target": torch.zeros(batch, encoder_cont.size(1)),
            "decoder_target": torch.zeros(batch, decoder_cont.size(1)),
            "decoder_time_idx": torch.arange(
                decoder_cont.size(1), dtype=torch.long
            )
            .unsqueeze(0)
            .expand(batch, -1),
            "groups": torch.zeros(batch, 1, dtype=torch.long),
        }
        out = self.model(x)
        if isinstance(out, dict):
            return out.get("prediction", next(iter(out.values())))
        if hasattr(out, "prediction"):
            return out.prediction
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


wrapped = PhysDistillEVWrapper(model).eval()

dummy = (
    torch.zeros(1, LOOKBACK, N_CONT),
    torch.zeros(1, HORIZON, N_CONT),
    torch.zeros(1, LOOKBACK, N_CAT, dtype=torch.long),
    torch.zeros(1, HORIZON, N_CAT, dtype=torch.long),
    torch.tensor([LOOKBACK], dtype=torch.long),
    torch.tensor([HORIZON], dtype=torch.long),
)

with torch.no_grad():
    torch_out = wrapped(*dummy)
print(f"  torch forward OK — output {tuple(torch_out.shape)}")

# Lightning 1.9 checkpoints predate the PyTorch 2.x hook dicts on nn.Module
for module in wrapped.modules():
    for attr in (
        "_state_dict_pre_hooks",
        "_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    ):
        if not hasattr(module, attr):
            setattr(module, attr, collections.OrderedDict())

ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    wrapped,
    dummy,
    str(ONNX_PATH),
    input_names=[
        "encoder_cont",
        "decoder_cont",
        "encoder_cat",
        "decoder_cat",
        "encoder_lengths",
        "decoder_lengths",
    ],
    output_names=["quantile_predictions"],
    dynamic_axes={
        "encoder_cont": {0: "batch"},
        "decoder_cont": {0: "batch"},
        "encoder_cat": {0: "batch"},
        "decoder_cat": {0: "batch"},
        "quantile_predictions": {0: "batch"},
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f"  exported -> {ONNX_PATH.name}")


# --------------------------------------------------------------------------
banner("6. Patch TopK scalar-K nodes")

import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

graph_model = onnx.load(str(ONNX_PATH))
patched = 0
for node in list(graph_model.graph.node):
    if node.op_type != "TopK":
        continue
    shape_name = f"topk_k_shape_topk_fix_{patched}"
    graph_model.graph.initializer.append(
        helper.make_tensor(shape_name, onnx.TensorProto.INT64, [1], [1])
    )
    reshaped = f"topk_k_reshaped_topk_fix_{patched}"
    insert_at = list(graph_model.graph.node).index(node)
    graph_model.graph.node.insert(
        insert_at,
        helper.make_node(
            "Reshape",
            [node.input[1], shape_name],
            [reshaped],
            name=f"Reshape_topk_fix_{patched}",
        ),
    )
    node.input[1] = reshaped
    patched += 1

onnx.save(graph_model, str(ONNX_PATH))
print(f"  patched {patched} TopK node(s)")


# --------------------------------------------------------------------------
banner("7. Validate — session construction, not onnx.checker")

import onnxruntime as ort  # noqa: E402

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

try:
    sess = ort.InferenceSession(
        str(ONNX_PATH), opts, providers=["CPUExecutionProvider"]
    )
except Exception as exc:
    print(f"  FAILED: graph does not load -> {exc}")
    sys.exit(1)

print("  session constructed — graph is executable")

feed = {
    "encoder_cont": dummy[0].numpy(),
    "decoder_cont": dummy[1].numpy(),
    "encoder_cat": dummy[2].numpy(),
    "decoder_cat": dummy[3].numpy(),
    "encoder_lengths": dummy[4].numpy(),
    "decoder_lengths": dummy[5].numpy(),
}
onnx_out = sess.run(None, feed)[0]
max_diff = float(np.abs(torch_out.numpy() - onnx_out).max())
print(f"  output shape       : {onnx_out.shape}")
print(f"  max |torch - onnx| : {max_diff:.8f}")

if max_diff >= 1e-4:
    print("  WARNING: divergence above tolerance — inspect before deploying")


# --------------------------------------------------------------------------
banner("8. Confirm the graph carries Jiaxing embeddings")

for init in onnx.load(str(ONNX_PATH)).graph.initializer:
    if "input_embeddings" in init.name:
        name = init.name.split(".")[-2]
        print(f"  {name:24} {numpy_helper.to_array(init).shape}")

print(f"\n  size on disk: {ONNX_PATH.stat().st_size / 1024:.0f} KB")
print("  Expect Location_Information (13, 7) and District_Name (3, 3).")
print("  If Station_Name appears, the wrong checkpoint was loaded.")