"""Frozen SigLIP embeddings. The encoder weights are not updated."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def feature_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Cannot read an embedding tensor from {type(output).__name__}.")


def load_siglip(model_id: str, device: torch.device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModel.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    scale = float(model.logit_scale.detach().float().exp().item())
    if scale <= 0 or scale != scale:
        raise RuntimeError(f"SigLIP logit scale is not usable: {scale}")
    return model, processor, scale


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _encode_pixels(model, processor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(feature_tensor(output).float(), dim=-1)


def embed_images(model, processor, paths: list[Path], device: torch.device, batch_size: int) -> torch.Tensor:
    """Embed each image and its horizontal flip, then average the two vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    encoded = []
    total = len(paths)
    for start in range(0, total, batch_size):
        batch_paths = paths[start : start + batch_size]
        originals = [_load_rgb(path) for path in batch_paths]
        views = originals + [ImageOps.mirror(image) for image in originals]
        features = _encode_pixels(model, processor, views, device)
        half = len(batch_paths)
        encoded.append(F.normalize(features[:half] + features[half:], dim=-1))
        done = min(start + batch_size, total)
        print(f"embedded {done}/{total} images", flush=True)
    if not encoded:
        raise RuntimeError("No images to embed.")
    return torch.cat(encoded, dim=0).cpu()


def embed_prompts(model, processor, prompts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    allowed = {"input_ids", "attention_mask"}
    kwargs = {key: value.to(device) for key, value in inputs.items() if key in allowed}
    with torch.inference_mode():
        output = model.get_text_features(**kwargs)
    return F.normalize(feature_tensor(output).float(), dim=-1).cpu()
