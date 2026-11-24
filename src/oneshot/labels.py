"""Class names as short phrases, and the prompt ensemble wrapped around them.

The phrases were written from the folder names before any test image was scored.
"""

from __future__ import annotations

VERBALIZER = {
    "Bedroom": "bedroom",
    "Coast": "coast",
    "Flower": "flower",
    "Forest": "forest",
    "Highway": "highway",
    "Industrial": "industrial area",
    "InsideCity": "city scene surrounded by buildings",
    "Kitchen": "kitchen",
    "LivingRoom": "living room",
    "Mountain": "mountain",
    "Office": "office",
    "OpenCountry": "open countryside",
    "Store": "store interior",
    "Street": "street",
    "Suburb": "suburban neighborhood",
    "TallBuilding": "tall building",
}

TEMPLATES = (
    "a photo of a {name}.",
    "a photo of the {name}.",
    "an image of a {name}.",
    "a {name} scene.",
)


def prompts_for(class_name: str) -> list[str]:
    try:
        name = VERBALIZER[class_name]
    except KeyError as exc:
        known = ", ".join(sorted(VERBALIZER))
        raise KeyError(f"No phrase for class {class_name!r}. Known classes: {known}") from exc
    return [template.format(name=name) for template in TEMPLATES]


def prompt_list(class_names: list[str]) -> list[str]:
    prompts: list[str] = []
    for class_name in class_names:
        prompts.extend(prompts_for(class_name))
    return prompts
