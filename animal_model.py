"""Model definition shared by the exporter and validation scripts."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# The pretrained ImageNet model has useful fine-grained animal knowledge, but
# Scrypted users generally want stable, searchable groups. Every ImageNet class
# belongs to exactly one output below, including the explicit non-animal class.
ANIMAL_GROUPS = OrderedDict(
    [
        ("fish", list(range(0, 7)) + list(range(389, 398))),
        ("bird", list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))),
        ("reptile_or_amphibian", list(range(25, 69))),
        ("invertebrate", list(range(69, 80)) + list(range(107, 127)) + list(range(300, 330))),
        ("marine_mammal", list(range(147, 151))),
        ("dog", list(range(151, 269))),
        ("wild_canine", list(range(269, 277))),
        ("fox", list(range(277, 281))),
        ("cat", list(range(281, 286))),
        ("wild_cat", list(range(286, 294))),
        ("bear", list(range(294, 298))),
        ("mongoose_or_meerkat", list(range(298, 300))),
        ("rabbit_or_hare", list(range(330, 333))),
        ("rodent", list(range(333, 339))),
        ("horse_or_zebra", list(range(339, 341))),
        ("pig_or_boar", list(range(341, 344))),
        ("hippopotamus", [344]),
        ("cattle_or_bison", list(range(345, 348))),
        ("sheep_or_goat", list(range(348, 351))),
        ("antelope_or_gazelle", list(range(351, 354))),
        ("camel_or_llama", list(range(354, 356))),
        ("weasel_otter_skunk_or_badger", list(range(356, 363))),
        ("armadillo", [363]),
        ("sloth", [364]),
        ("primate", list(range(365, 385))),
        ("elephant", list(range(385, 387))),
        ("panda", list(range(387, 389))),
    ]
)

OUTPUT_LABELS = list(ANIMAL_GROUPS) + ["not_animal"]


def validate_groups() -> None:
    flattened = [index for indices in ANIMAL_GROUPS.values() for index in indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Animal groups overlap")
    if any(index < 0 or index >= 1000 for index in flattened):
        raise ValueError("Animal group contains an invalid ImageNet index")


class AnimalClassifier(nn.Module):
    """Aggregate MobileNet ImageNet probabilities into broad animal groups.

    The module emits log probabilities. Scrypted's custom ResNet classifier
    applies softmax to model output, which recovers the aggregated probabilities.
    """

    def __init__(self, weights=MobileNet_V3_Small_Weights.DEFAULT):
        super().__init__()
        validate_groups()
        self.backbone = mobilenet_v3_small(weights=weights)
        self.backbone.eval()

        used = {index for indices in ANIMAL_GROUPS.values() for index in indices}
        groups = list(ANIMAL_GROUPS.values()) + [[index for index in range(1000) if index not in used]]
        aggregation = torch.zeros((len(groups), 1000), dtype=torch.float32)
        for group_index, indices in enumerate(groups):
            aggregation[group_index, indices] = 1
        self.register_buffer("aggregation", aggregation)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.backbone(image), dim=1)
        grouped = torch.matmul(probabilities, self.aggregation.transpose(0, 1))
        return torch.log(grouped + 1e-12)
