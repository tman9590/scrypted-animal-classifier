#!/usr/bin/env python3
"""Export the general animal classifier for Scrypted inference backends."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from animal_model import AnimalClassifier, OUTPUT_LABELS


ROOT = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 3, 224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def config(files: list[str]) -> dict:
    return {
        "input_shape": list(INPUT_SHAPE),
        "model": "resnet",
        "mean": MEAN,
        "std": STD,
        "files": files,
        "labels": {str(index): label for index, label in enumerate(OUTPUT_LABELS)},
    }


def write_config(directory: Path, files: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config(files), indent=2) + "\n")


def export_onnx(model: torch.nn.Module, example: torch.Tensor) -> None:
    directory = ROOT / "models" / "onnx"
    directory.mkdir(parents=True, exist_ok=True)
    filename = "animal-mobilenet-v3-small.onnx"
    torch.onnx.export(
        model,
        example,
        directory / filename,
        input_names=["image"],
        output_names=["animal_log_probabilities"],
        opset_version=17,
        dynamo=False,
    )
    write_config(directory, [filename])


def export_openvino(model: torch.nn.Module, example: torch.Tensor) -> None:
    import openvino as ov

    directory = ROOT / "models" / "openvino"
    directory.mkdir(parents=True, exist_ok=True)
    converted = ov.convert_model(model, example_input=example)
    ov.save_model(converted, directory / "animal.xml", compress_to_fp16=False)
    write_config(directory, ["animal.xml", "animal.bin"])


def export_coreml(model: torch.nn.Module, example: torch.Tensor) -> None:
    import coremltools as ct

    directory = ROOT / "models" / "coreml"
    package = directory / "animal.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    directory.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=INPUT_SHAPE)],
        convert_to="mlprogram",
    )
    converted.save(package)
    files = [str(path.relative_to(directory)) for path in sorted(package.rglob("*")) if path.is_file()]
    write_config(directory, files)


def export_ncnn(model: torch.nn.Module, example: torch.Tensor) -> None:
    directory = ROOT / "models" / "ncnn"
    directory.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example)
    torchscript = directory / "animal.pt"
    traced.save(str(torchscript))

    import subprocess

    pnnx = shutil.which("pnnx") or str(Path(sys.executable).with_name("pnnx"))
    subprocess.run(
        [pnnx, str(torchscript), f"inputshape={list(INPUT_SHAPE)}", "fp16=1"],
        cwd=directory,
        check=True,
    )
    torchscript.unlink()
    generated_param = directory / "animal.ncnn.param"
    generated_bin = directory / "animal.ncnn.bin"
    if not generated_param.exists() or not generated_bin.exists():
        raise RuntimeError("pnnx did not create the expected NCNN model files")
    write_config(directory, [generated_param.name, generated_bin.name])
    for generated in (
        directory / "animal.pnnx.param",
        directory / "animal.pnnx.bin",
        directory / "animal.pnnx.onnx",
        directory / "animal_pnnx.py",
        directory / "animal_ncnn.py",
    ):
        generated.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["onnx", "openvino", "coreml", "ncnn"],
        default=["onnx", "openvino", "coreml", "ncnn"],
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    model = AnimalClassifier().eval()
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    exporters = {
        "onnx": export_onnx,
        "openvino": export_openvino,
        "coreml": export_coreml,
        "ncnn": export_ncnn,
    }
    for backend in args.backends:
        print(f"Exporting {backend}…")
        exporters[backend](model, example)


if __name__ == "__main__":
    main()
