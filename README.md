# Scrypted Animal Classifier

A local, hardware-accelerated image classifier for Scrypted. It groups a
pretrained MobileNet V3 model into camera-friendly animal categories and has an
explicit `not_animal` result to reduce false animal events.

The repository also retains the original 450-species bird exporter under
`legacy/` and its original model assets in Git history. The current default
model is the broader animal classifier.

## Classes

The model reports 27 animal groups plus `not_animal`:

`fish`, `bird`, `reptile_or_amphibian`, `invertebrate`, `marine_mammal`,
`dog`, `wild_canine`, `fox`, `cat`, `wild_cat`, `bear`,
`mongoose_or_meerkat`, `rabbit_or_hare`, `rodent`, `horse_or_zebra`,
`pig_or_boar`, `hippopotamus`, `cattle_or_bison`, `sheep_or_goat`,
`antelope_or_gazelle`, `camel_or_llama`,
`weasel_otter_skunk_or_badger`, `armadillo`, `sloth`, `primate`, `elephant`,
and `panda`.

This is an image classifier, not a full-frame object detector. It performs best
when Scrypted passes it a crop centered on a detected animal. It does not return
bounding boxes. The underlying ImageNet model has no dedicated raccoon class;
raccoons may be confused with badgers, pandas, or other small mammals.

## Install in Scrypted

1. Install the object-detection backend that matches the Scrypted server:
   **CoreML** for Apple Silicon, **OpenVINO** for Intel/AMD, **ONNX** for an
   NVIDIA setup, or **NCNN** where that backend is available.
2. Open that backend plugin and choose **Create Device**.
3. Set the model name to `Animal Classifier`.
4. Use this repository URL as the model URL:
   `https://github.com/tman9590/scrypted-animal-classifier`
5. In Scrypted NVR, select the created classifier where an image classifier is
   requested. Use a confidence threshold of at least 0.65 initially and tune it
   with footage from the actual cameras.

Scrypted resolves the repository URL to `models/<backend>/config.json` and
downloads only the files listed by that config. Inference stays on the Scrypted
server after the model is downloaded.

## Privacy and performance

Classification is local. Installing the model downloads model files from
GitHub, but camera images are not sent to this repository or an external model
service. MobileNet V3 Small uses a 224×224 RGB input and is intentionally small
enough for frequent classification on typical Scrypted hosts.

## Build

Use Python 3.11 or newer. The exporter downloads the public pretrained
MobileNet V3 Small weights on first run.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-build.txt
.venv/bin/python build_animal_models.py
.venv/bin/python validate_exports.py
.venv/bin/python -m unittest discover -s tests -v
```

Export one or more backends with `--backends`, for example:

```sh
.venv/bin/python build_animal_models.py --backends onnx openvino
```

## Model behavior

The pretrained network produces 1,000 ImageNet probabilities. The wrapper sums
related fine-grained classes into broad groups, then emits their log
probabilities. Scrypted's custom-classifier softmax recovers those exact grouped
probabilities. All unused ImageNet classes are combined into `not_animal`, so
the animal probabilities and `not_animal` probability sum to one.

The output is useful for notifications and event labels, but it is not a
wildlife survey instrument. Camera angle, night vision, motion blur, small
subjects, and regional species all affect accuracy. Verify important results
against recorded footage.

## License and attribution

Code in this repository is provided under the Apache-2.0 license. The classifier
uses torchvision's pretrained MobileNet V3 Small weights; review the upstream
torchvision documentation for training-data and model limitations.
