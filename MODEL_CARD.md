# Model card: Scrypted Animal Classifier

## Intended use

This model assigns a single animal-centered image crop to one of 27 broad
animal groups or `not_animal`. It is intended as a second-stage classifier for
Scrypted camera events. It is not intended to locate animals, produce bounding
boxes, identify individual pets, or support scientific population counts.

## Architecture and data

- Backbone: torchvision MobileNet V3 Small with public pretrained weights.
- Source label space: ImageNet-1K's 1,000 classes.
- Input: one normalized 224×224 RGB image.
- Output: 28 log probabilities. Scrypted applies softmax to recover the group
  probabilities.
- Aggregation: each ImageNet class belongs to exactly one output group. Classes
  outside the animal groups are summed into `not_animal`.

No additional camera footage, private imagery, or user data was used to create
the model.

## Validation

The checked-in exports are compared against the PyTorch reference on a seeded
input. Maximum observed probability differences during release validation:

| Backend | Maximum difference |
|---|---:|
| ONNX Runtime | 0.000001 |
| OpenVINO | 0.028401 |
| CoreML | 0.006642 |
| NCNN FP16 | 0.038963 |

Two public smoke-test images were also checked: the PyTorch Hub dog image
classified as `dog` with 0.9866 probability, and the repository's American
goldfinch image classified as `bird` with 0.9993 probability. These examples
are sanity checks, not an accuracy benchmark.

## Limitations

- The model inherits the biases and gaps of ImageNet-1K.
- ImageNet-1K has no dedicated raccoon class and no general deer class.
  Raccoons may resemble badgers or pandas; deer may resemble antelope or other
  hoofed animals.
- Broad groups deliberately trade species detail for stability. The specialist
  legacy bird model is more suitable when bird species are the only target.
- Infrared night images, severe motion blur, partial animals, tiny subjects,
  unusual viewpoints, and multiple animals in one crop can lower confidence or
  produce the wrong group.
- `not_animal` is a necessary rejection class, but a threshold should still be
  tuned on representative footage.

## Operational guidance

Start with a 0.65 confidence threshold. Review false positives and missed
events from each physical camera before using a label in automations. Do not
use the output for safety-critical decisions.

