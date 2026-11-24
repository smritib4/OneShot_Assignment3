"""Closed-form one-shot classifier in SigLIP's embedding space.

Each class gets one weight vector: the normalized sum of its text embedding
and the embedding of its single labeled photo. No other labeled photo is read,
and nothing in this file accepts a validation or test label.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1)


def average_prompt_features(prompt_features: torch.Tensor, n_classes: int, n_prompts: int) -> torch.Tensor:
    if prompt_features.shape[0] != n_classes * n_prompts:
        raise ValueError(
            f"Expected {n_classes * n_prompts} prompt embeddings, got {prompt_features.shape[0]}."
        )
    pooled = prompt_features.float().view(n_classes, n_prompts, -1).mean(dim=1)
    return l2_normalize(pooled)


def class_weights(text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
    if text_features.shape != visual_features.shape:
        raise ValueError(
            f"Text and visual features differ: {tuple(text_features.shape)} vs {tuple(visual_features.shape)}."
        )
    return l2_normalize(text_features.float() + visual_features.float())


def score(image_features: torch.Tensor, weights: torch.Tensor, logit_scale: float) -> torch.Tensor:
    return float(logit_scale) * image_features.float() @ weights.float().T


def predict(class_scores: torch.Tensor) -> torch.Tensor:
    return class_scores.argmax(dim=-1)


def accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.to(prediction.device)
    return float((prediction == labels).float().mean().item())


def confusion(prediction: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for true_label, predicted in zip(labels.tolist(), prediction.tolist()):
        matrix[int(true_label), int(predicted)] += 1
    return matrix
