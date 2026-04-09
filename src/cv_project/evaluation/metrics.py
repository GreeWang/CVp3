from __future__ import annotations

import cv2
import numpy as np


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 0
    gt = gt_mask > 0
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(pred & gt)
    return float(intersection) / float(union)


def jaccard_mean(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Prediction and ground-truth sequences must have the same length.")
    scores = [mask_iou(pred, gt) for pred, gt in zip(pred_masks, gt_masks)]
    return float(np.mean(scores)) if scores else 0.0


def jaccard_recall(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], threshold: float = 0.5) -> float:
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Prediction and ground-truth sequences must have the same length.")
    scores = [mask_iou(pred, gt) for pred, gt in zip(pred_masks, gt_masks)]
    return float(np.mean([score >= threshold for score in scores])) if scores else 0.0


def psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    diff = image_a.astype(np.float32) - image_b.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    if a.ndim == 3:
        scores = [ssim(a[..., channel], b[..., channel]) for channel in range(a.shape[2])]
        return float(np.mean(scores))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_a = cv2.GaussianBlur(a, kernel, sigma)
    mu_b = cv2.GaussianBlur(b, kernel, sigma)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, kernel, sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, kernel, sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, kernel, sigma) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))
