"""Offline checks for local-embedding device/runtime resolution.

Plain asserts, no framework, no torch import -- resolve_device() only touches
torch through detect_device(), and only for device: auto.

    .venv/bin/python test_device.py
"""

import pipeline
from pipeline import default_batch_size, resolve_device


def resolve(**cfg):
    return resolve_device(cfg)


def test_torch_runtime():
    # auto asks detect_device(); stub it so this stays offline and torch-free
    original = pipeline.detect_device
    pipeline.detect_device = lambda: "cuda"
    try:
        assert resolve() == ("cuda", {}), "empty config must behave as torch/auto"
        assert resolve(runtime="torch", device="auto") == ("cuda", {})
    finally:
        pipeline.detect_device = original

    assert resolve(runtime="torch", device="cpu") == ("cpu", {})
    assert resolve(runtime="torch", device="xpu") == ("xpu", {})
    assert resolve(runtime="torch", device="mps") == ("mps", {})


def test_openvino_runtime():
    # torch stays on the cpu; the accelerator rides along as a model kwarg
    assert resolve(runtime="openvino", device="auto") == ("cpu", {"device": "AUTO"})
    assert resolve(runtime="openvino", device="xpu") == ("cpu", {"device": "GPU"})
    assert resolve(runtime="openvino", device="npu") == ("cpu", {"device": "NPU"})
    assert resolve(runtime="openvino", device="cpu") == ("cpu", {"device": "CPU"})


def test_onnx_runtime():
    assert resolve(runtime="onnx", device="auto") == ("cpu", {}), "auto must not pin a provider"
    assert resolve(runtime="onnx", device="cuda") == ("cpu", {"provider": "CUDAExecutionProvider"})
    # ROCm is its own provider here even though torch calls that device "cuda"
    assert resolve(runtime="onnx", device="rocm") == ("cpu", {"provider": "ROCMExecutionProvider"})
    assert resolve(runtime="onnx", device="xpu") == ("cpu", {"provider": "OpenVINOExecutionProvider"})


def test_rejects_mismatches():
    def rejects(msg_fragment, **cfg):
        try:
            resolve(**cfg)
        except RuntimeError as e:
            assert msg_fragment in str(e), f"unhelpful message: {e}"
        else:
            raise AssertionError(f"expected a RuntimeError for {cfg}")

    rejects("openvino", runtime="openvino", device="cuda")  # OpenVINO has no cuda
    rejects("openvino", runtime="openvino", device="mps")
    rejects("torch", runtime="torch", device="npu")         # npu is openvino-only
    rejects("torch", runtime="torch", device="gpu")         # not a device name
    rejects("Unknown embedding runtime", runtime="tensorrt", device="auto")


def test_batch_sizes():
    assert default_batch_size("cuda:0") == 64, "an indexed cuda device is still cuda"
    assert default_batch_size("cuda") == 64
    assert default_batch_size("xpu") == 64
    assert default_batch_size("rocm") == 64
    assert default_batch_size("cpu") == 16
    assert default_batch_size("npu") == 16, "unknown families fall back to the safe size"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all device tests passed")
