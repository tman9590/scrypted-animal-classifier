#!/usr/bin/env python3
"""Compare exported backends with the PyTorch reference implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from animal_model import AnimalClassifier, OUTPUT_LABELS


ROOT = Path(__file__).resolve().parent


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def check(name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape}, expected {expected.shape}")
    difference = float(np.max(np.abs(softmax(actual) - softmax(expected))))
    if not np.isfinite(actual).all() or difference > tolerance:
        raise AssertionError(f"{name}: maximum probability difference {difference} exceeds {tolerance}")
    print(f"OK {name}: shape={actual.shape}, max probability difference={difference:.6f}")


def validate_configs() -> None:
    expected_labels = {str(index): label for index, label in enumerate(OUTPUT_LABELS)}
    for backend in ("onnx", "openvino", "coreml", "ncnn"):
        directory = ROOT / "models" / backend
        config = json.loads((directory / "config.json").read_text())
        if config["labels"] != expected_labels:
            raise AssertionError(f"{backend}: labels differ")
        for filename in config["files"]:
            if not (directory / filename).is_file():
                raise AssertionError(f"{backend}: missing {filename}")
        print(f"OK {backend}: config lists {len(config['files'])} existing file(s)")


def main() -> None:
    torch.manual_seed(7)
    input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    reference_model = AnimalClassifier().eval()
    with torch.no_grad():
        reference = reference_model(input_tensor).numpy()

    import onnx
    import onnxruntime

    onnx_path = ROOT / "models" / "onnx" / "animal-mobilenet-v3-small.onnx"
    onnx.checker.check_model(onnx.load(onnx_path))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_result = session.run(None, {session.get_inputs()[0].name: input_tensor.numpy()})[0]
    check("ONNX Runtime", onnx_result, reference, 2e-4)

    import openvino as ov

    compiled = ov.Core().compile_model(str(ROOT / "models" / "openvino" / "animal.xml"), "CPU")
    openvino_result = next(iter(compiled({0: input_tensor.numpy()}).values()))
    check("OpenVINO", openvino_result, reference, 4e-2)

    import coremltools as ct

    coreml = ct.models.MLModel(str(ROOT / "models" / "coreml" / "animal.mlpackage"))
    coreml_result = next(iter(coreml.predict({"image": input_tensor.numpy()}).values()))
    check("CoreML", coreml_result, reference, 1e-2)

    import ncnn

    with ncnn.Net() as net:
        net.load_param(str(ROOT / "models" / "ncnn" / "animal.ncnn.param"))
        net.load_model(str(ROOT / "models" / "ncnn" / "animal.ncnn.bin"))
        with net.create_extractor() as extractor:
            extractor.input("in0", ncnn.Mat(input_tensor.squeeze(0).numpy()).clone())
            result_code, output = extractor.extract("out0")
            if result_code != 0:
                raise AssertionError(f"NCNN inference failed with code {result_code}")
            ncnn_result = np.array(output)[None, :]
    check("NCNN", ncnn_result, reference, 4e-2)

    validate_configs()


if __name__ == "__main__":
    main()
