"""Long stable line geometry; independent of surface color and texture."""

import cv2
import numpy as np


def extract_lines(image, mask):
    h, w = image.shape
    diagonal = np.hypot(w, h)
    edges = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 1), 35, 100)
    edges[mask == 0] = 0
    found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40,
                            minLineLength=diagonal * 0.18, maxLineGap=diagonal * 0.025)
    if found is None:
        return np.empty((0, 4), np.float32)
    candidates = found[:, 0].astype(np.float32)
    lengths = np.linalg.norm(candidates[:, 2:] - candidates[:, :2], axis=1)
    kept = []
    for segment in candidates[np.argsort(-lengths)]:
        points = np.linspace(segment[:2], segment[2:], 40).astype(int)
        if np.mean(mask[points[:, 1], points[:, 0]] > 0) < 0.85:
            continue
        # Do not count both edges of one thick boundary as independent evidence.
        if any(line_distance(segment, old, diagonal * 0.009, 2)[0] for old in kept):
            continue
        kept.append(segment)
    return np.array(kept[:30], np.float32).reshape(-1, 4)


def line_distance(a, b, tolerance, max_angle):
    pa, pb = a.reshape(2, 2), b.reshape(2, 2)
    va, vb = pa[1] - pa[0], pb[1] - pb[0]
    la, lb = np.linalg.norm(va), np.linalg.norm(vb)
    direction = va / max(la, 1e-6)
    cosine = np.clip(abs(np.dot(va, vb)) / max(la * lb, 1e-6), 0, 1)
    angle = np.degrees(np.arccos(cosine))
    normal = np.array([-direction[1], direction[0]])
    distance = float(np.max(np.abs((pb - pa[0]) @ normal)))
    projections = sorted((pb - pa[0]) @ direction)
    overlap = max(0, min(la, projections[1]) - max(0, projections[0]))
    valid = angle <= max_angle and distance <= tolerance and overlap >= 0.6 * min(la, lb)
    return valid, distance + angle


def diverse(lines):
    if len(lines) < 3:
        return False
    vectors = lines[:, 2:] - lines[:, :2]
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return bool(np.any(np.abs(vectors @ vectors.T) < np.cos(np.deg2rad(20))))


def compare_lines(reference, current):
    h, w = reference.image.shape
    image = cv2.resize(current.image, (w, h))
    mask = cv2.resize(current.mask, (w, h), interpolation=cv2.INTER_NEAREST)
    a = extract_lines(reference.image, reference.mask)
    b = extract_lines(image, mask)
    pairs = []
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            valid, distance = line_distance(first, second, np.hypot(w, h) * 0.004, 1.5)
            if valid:
                pairs.append((distance, i, j))
    used_a, used_b = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    ratio = min(len(used_a) / max(len(a), 1), len(used_b) / max(len(b), 1))
    # Parallel lines alone cannot rule out movement along their direction.
    stationary = len(used_a) >= 3 and ratio >= 0.65 and diverse(a[sorted(used_a)])
    mismatch = diverse(a) and diverse(b) and ratio < 0.15
    return {"reference_lines": len(a), "current_lines": len(b),
            "matched_lines": len(used_a), "line_match_ratio": ratio}, stationary, mismatch
