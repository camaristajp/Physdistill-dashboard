"""
core/engine.py

One ONNX Runtime session, created once and reused. Single-threaded and
sequential, matching the paper's Raspberry Pi benchmark conditions.

Building a session per request turns 2 ms of inference into a 300 ms stall,
so the session is module-level and the Streamlit pages hold it in
st.cache_resource.
"""

import time
from functools import lru_cache

import numpy as np
import onnxruntime as ort

from core import config

_WARMUP_PASSES = 10


@lru_cache(maxsize=1)
def get_session():
    config.require(
        config.ONNX_PATH,
        "Run scripts/convert_physdistill_ev_jiaxing.py",
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = config.ORT_INTRA_OP_THREADS
    options.inter_op_num_threads = config.ORT_INTER_OP_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(config.ONNX_PATH), options, providers=["CPUExecutionProvider"]
    )

    # Warm the graph so the first user-facing forecast is not the slow one.
    shapes = {i.name: i.shape for i in session.get_inputs()}
    dummy = {
        "encoder_cont": np.zeros((1, config.ENCODER_LENGTH, shapes["encoder_cont"][2]), np.float32),
        "decoder_cont": np.zeros((1, config.HORIZON, shapes["decoder_cont"][2]), np.float32),
        "encoder_cat": np.zeros((1, config.ENCODER_LENGTH, shapes["encoder_cat"][2]), np.int64),
        "decoder_cat": np.zeros((1, config.HORIZON, shapes["decoder_cat"][2]), np.int64),
        "encoder_lengths": np.array([config.ENCODER_LENGTH], np.int64),
        "decoder_lengths": np.array([config.HORIZON], np.int64),
    }
    for _ in range(_WARMUP_PASSES):
        session.run(None, dummy)

    return session


def input_signature():
    """Channel counts, for the System page and for validating window output."""
    session = get_session()
    return {i.name: list(i.shape) for i in session.get_inputs()}


def run(feed):
    """Single forward pass. Returns raw log-space quantiles and latency in ms."""
    session = get_session()
    start = time.perf_counter()
    output = session.run(None, feed)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output[0], elapsed_ms


def benchmark(feed, passes=200, warmup=10):
    """Latency profile under the paper's measurement protocol."""
    session = get_session()
    for _ in range(warmup):
        session.run(None, feed)

    timings = np.empty(passes, dtype=np.float64)
    for i in range(passes):
        start = time.perf_counter()
        session.run(None, feed)
        timings[i] = (time.perf_counter() - start) * 1000.0

    return {
        "mean_ms": float(timings.mean()),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "std_ms": float(timings.std()),
        "passes": int(passes),
    }
