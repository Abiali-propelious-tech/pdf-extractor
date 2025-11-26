#!/usr/bin/env python3
"""
Group image and text blocks by proximity from extractor JSON.

Usage:
  python group_blocks.py <input_json> [--out <out_json>] [--threshold 5]

Produces an output JSON with an added `groups` array per page. Each group
contains `id`, `bbox`, `images` (ids), and `texts` (ids).
"""
import json
import argparse
import os
import math
from typing import List, Tuple


def rect_distance(a: List[float], b: List[float]) -> float:
    """Return minimal Euclidean distance between two axis-aligned rectangles.

    Rectangles are [x0,y0,x1,y1]. If they overlap, distance is 0.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # horizontal gap
    if ax1 < bx0:
        dx = bx0 - ax1
    elif bx1 < ax0:
        dx = ax0 - bx1
    else:
        dx = 0
    # vertical gap
    if ay1 < by0:
        dy = by0 - ay1
    elif by1 < ay0:
        dy = ay0 - by1
    else:
        dy = 0
    return math.hypot(dx, dy)


def rects_overlap(a: List[float], b: List[float]) -> bool:
    """Return True if two rects [x0,y0,x1,y1] overlap (non-empty intersection)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        if x not in self.p:
            self.p[x] = x
            return x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        self.p[rb] = ra


def detect_page_layout(page: dict) -> str:
    """Analyze page structure and return layout type.

    Returns one of:
    - 'catalog_grid': Wide images in grid, catalog-style with captions
    - 'table_grid': Regular grid of similar-sized images/text blocks
    - 'multi_column': Multiple text columns with images
    - 'sidebar': Images on one side, text on the other
    - 'centered_cluster': Images grouped in center, text surrounds them
    - 'vertical_flow': Top-to-bottom flow, images interspersed with text
    - 'horizontal_strips': Horizontal bands of content
    - 'scattered_dense': Many small elements scattered densely
    - 'scattered_sparse': Few elements spread out
    - 'full_page_single': One or few large elements dominating page
    - 'mixed_density': Mixed areas of dense and sparse content
    """
    images = page.get("images", [])
    texts = page.get("texts", [])

    if not images or not texts:
        return "scattered"

    # Get page dimensions from all elements
    all_x0 = []
    all_y0 = []
    all_x1 = []
    all_y1 = []

    for img in images:
        bbox = img.get("bbox", [])
        if len(bbox) == 4:
            all_x0.append(bbox[0])
            all_y0.append(bbox[1])
            all_x1.append(bbox[2])
            all_y1.append(bbox[3])

    for txt in texts:
        bbox = txt.get("bbox", [])
        if len(bbox) == 4:
            all_x0.append(bbox[0])
            all_y0.append(bbox[1])
            all_x1.append(bbox[2])
            all_y1.append(bbox[3])

    if not all_x0:
        return "scattered"

    page_left = min(all_x0)
    page_right = max(all_x1)
    page_top = min(all_y0)
    page_bottom = max(all_y1)
    page_width = page_right - page_left
    page_height = page_bottom - page_top

    if page_width < 1 or page_height < 1:
        return "scattered"

    # Analyze image positions
    img_centers_x = []
    img_centers_y = []
    img_widths = []

    for img in images:
        bbox = img.get("bbox", [])
        if len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            w = bbox[2] - bbox[0]
            img_centers_x.append(cx)
            img_centers_y.append(cy)
            img_widths.append(w)

    if not img_centers_x:
        # No images - check text layout
        if len(texts) == 0:
            return "full_page_single"
        return "vertical_flow"

    # Calculate statistics
    num_images = len(images)
    num_texts = len(texts)
    total_elements = num_images + num_texts

    img_span_x = (
        max(img_centers_x) - min(img_centers_x) if len(img_centers_x) > 1 else 0
    )
    img_span_y = (
        max(img_centers_y) - min(img_centers_y) if len(img_centers_y) > 1 else 0
    )
    avg_img_width = sum(img_widths) / len(img_widths)

    # Element density
    page_area = page_width * page_height
    density = total_elements / (page_area / 10000) if page_area > 0 else 0

    # Pattern 1: FULL_PAGE_SINGLE - Very few large elements
    if total_elements <= 3 and avg_img_width > page_width * 0.5:
        return "full_page_single"

    # Pattern 2: CATALOG_GRID - Wide images in grid, catalog-style
    if (
        num_images >= 4
        and img_span_x > page_width * 0.6
        and avg_img_width > page_width * 0.25
    ):
        # Check for grid-like spacing
        img_widths_similar = (max(img_widths) - min(img_widths)) < avg_img_width * 0.3
        if img_widths_similar:
            return "catalog_grid"

    # Pattern 3: TABLE_GRID - Regular grid of similar-sized elements
    if num_images >= 6:
        img_widths_similar = (max(img_widths) - min(img_widths)) < avg_img_width * 0.4
        # Check if images form rows/columns
        y_positions = sorted(
            set(round(cy / 50) * 50 for cy in img_centers_y)
        )  # Bucket by ~50px
        x_positions = sorted(set(round(cx / 50) * 50 for cx in img_centers_x))
        if len(y_positions) >= 2 and len(x_positions) >= 2 and img_widths_similar:
            return "table_grid"

    # Pattern 4: SIDEBAR - Images on one side, text on other
    left_third = page_left + page_width * 0.35
    right_third = page_left + page_width * 0.65

    images_on_left = sum(1 for cx in img_centers_x if cx < left_third)
    images_on_right = sum(1 for cx in img_centers_x if cx > right_third)

    if (
        images_on_left > len(img_centers_x) * 0.75
        or images_on_right > len(img_centers_x) * 0.75
    ):
        return "sidebar"

    # Pattern 5: CENTERED_CLUSTER - Images concentrated in center
    center_x = page_left + page_width / 2.0
    center_y = page_top + page_height / 2.0
    center_margin_x = page_width * 0.3
    center_margin_y = page_height * 0.3

    images_in_center = sum(
        1
        for cx, cy in zip(img_centers_x, img_centers_y)
        if abs(cx - center_x) < center_margin_x and abs(cy - center_y) < center_margin_y
    )

    if images_in_center > len(img_centers_x) * 0.6:
        return "centered_cluster"

    # Pattern 6: MULTI_COLUMN - Check for columnar text layout
    if num_texts > num_images * 3:
        # Analyze text X positions to detect columns
        text_x_centers = [
            (txt.get("bbox", [0, 0, 0, 0])[0] + txt.get("bbox", [0, 0, 0, 0])[2]) / 2.0
            for txt in texts
            if len(txt.get("bbox", [])) == 4
        ]
        if text_x_centers:
            # Bucket texts by X position
            x_buckets = {}
            bucket_size = page_width / 10
            for x in text_x_centers:
                bucket = int(x / bucket_size)
                x_buckets[bucket] = x_buckets.get(bucket, 0) + 1
            # If we have 2+ distinct columns with many texts
            significant_buckets = [
                b for b, count in x_buckets.items() if count > num_texts * 0.15
            ]
            if len(significant_buckets) >= 2:
                return "multi_column"

    # Pattern 7: VERTICAL_FLOW - Top-to-bottom flow
    if img_span_y > page_height * 0.5 and img_span_x < page_width * 0.4:
        # Images stacked vertically
        return "vertical_flow"

    # Pattern 8: HORIZONTAL_STRIPS - Content in horizontal bands
    if img_span_x > page_width * 0.7 and img_span_y < page_height * 0.3:
        return "horizontal_strips"

    # Pattern 9: SCATTERED_DENSE vs SCATTERED_SPARSE
    if density > 5.0:  # High density
        return "scattered_dense"
    elif density < 1.5:  # Low density
        return "scattered_sparse"

    # Pattern 10: MIXED_DENSITY - Check for clustering
    if num_images >= 4:
        # Calculate variance in distances between images
        distances = []
        for i in range(len(img_centers_x)):
            for j in range(i + 1, len(img_centers_x)):
                dx = img_centers_x[i] - img_centers_x[j]
                dy = img_centers_y[i] - img_centers_y[j]
                distances.append(math.sqrt(dx * dx + dy * dy))
        if distances:
            avg_dist = sum(distances) / len(distances)
            variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
            if variance > avg_dist * avg_dist:  # High variance = mixed density
                return "mixed_density"

    return "scattered_sparse"


def group_page(page: dict, threshold: float = 5.0, mode: str = "expand") -> List[dict]:
    """Group image and text blocks on a page.

    Args:
      page: page dict with 'images' and 'texts' lists, each having 'bbox' as [x0,y0,x1,y1].
      threshold: proximity threshold in the same units as bboxes. For 'expand' mode
                 this is used as a padding applied to each bbox before testing overlap.
      mode: 'expand' (default) — expand boxes by threshold and group overlaps;
            'distance' — use minimal edge-to-edge Euclidean distance (legacy behavior).

    Returns: list of groups where each group contains full image/text objects.
    """

    # STEP 0: Detect page layout pattern FIRST.
    # Include all extracted images and texts for layout detection. Filtering
    # out blocks based on `blocked` can remove important signals (for
    # example many detail text blocks with missing sizes) and lead to wrong
    # layout decisions. Use the raw extracted elements so layout detection is
    # robust to downstream blocking heuristics.
    cleaned_page_for_detection = {
        "images": [im for im in page.get("images", [])],
        "texts": [t for t in page.get("texts", [])],
    }
    # copy dimensions if available
    if page.get("width"):
        cleaned_page_for_detection["width"] = page.get("width")
    if page.get("height"):
        cleaned_page_for_detection["height"] = page.get("height")

    layout_type = detect_page_layout(cleaned_page_for_detection)
    page["_detected_layout"] = layout_type

    # control debug output with environment variable
    try:
        DEBUG_GROUPER = os.getenv("PDF_GROUP_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
    except Exception:
        DEBUG_GROUPER = False

    # Configure algorithm parameters based on detected layout
    if layout_type == "catalog_grid":
        # Wide catalog: aggressive horizontal, each image gets nearby text
        radius_multiplier = 3.5
        directional_multiplier = 6.0
        align_frac = 0.45
        large_multiplier = 7.0
        MAX_TEXTS_PER_GROUP = 12
        MAX_CHARS_PER_GROUP = 1500
        text_text_threshold = threshold * 0.8  # Tight text grouping

    elif layout_type == "table_grid":
        # Regular grid: very conservative, each cell separate
        radius_multiplier = 1.2
        directional_multiplier = 2.0
        align_frac = 0.8
        large_multiplier = 2.5
        MAX_TEXTS_PER_GROUP = 4
        MAX_CHARS_PER_GROUP = 400
        text_text_threshold = threshold * 0.6

    elif layout_type == "multi_column":
        # Columns: respect column boundaries, vertical grouping
        radius_multiplier = 1.5
        directional_multiplier = 3.0
        align_frac = 0.75
        large_multiplier = 3.5
        MAX_TEXTS_PER_GROUP = 8
        MAX_CHARS_PER_GROUP = 1000
        text_text_threshold = threshold * 1.2  # Allow more text grouping within column

    elif layout_type == "sidebar":
        # Side-by-side: wide horizontal search across divide
        radius_multiplier = 5.0
        directional_multiplier = 8.0
        align_frac = 0.35
        large_multiplier = 10.0
        MAX_TEXTS_PER_GROUP = 10
        MAX_CHARS_PER_GROUP = 1200
        text_text_threshold = threshold

    elif layout_type == "centered_cluster":
        # Center focus: conservative, small groups
        radius_multiplier = 1.5
        directional_multiplier = 2.0
        align_frac = 0.7
        large_multiplier = 3.0
        MAX_TEXTS_PER_GROUP = 5
        MAX_CHARS_PER_GROUP = 600
        text_text_threshold = threshold * 0.7

    elif layout_type == "vertical_flow":
        # Top-to-bottom: prioritize vertical proximity
        radius_multiplier = 2.5
        directional_multiplier = 4.0
        align_frac = 0.5
        large_multiplier = 5.0
        MAX_TEXTS_PER_GROUP = 8
        MAX_CHARS_PER_GROUP = 1000
        text_text_threshold = threshold * 1.0

    elif layout_type == "horizontal_strips":
        # Horizontal bands: prioritize horizontal proximity
        radius_multiplier = 4.0
        directional_multiplier = 7.0
        align_frac = 0.4
        large_multiplier = 8.0
        MAX_TEXTS_PER_GROUP = 10
        MAX_CHARS_PER_GROUP = 1200
        text_text_threshold = threshold * 1.5

    elif layout_type == "scattered_dense":
        # Dense scatter: tight grouping, small groups
        radius_multiplier = 1.3
        directional_multiplier = 2.0
        align_frac = 0.75
        large_multiplier = 2.5
        MAX_TEXTS_PER_GROUP = 6
        MAX_CHARS_PER_GROUP = 700
        text_text_threshold = threshold * 0.8

    elif layout_type == "scattered_sparse":
        # Sparse scatter: wider search radius
        radius_multiplier = 3.0
        directional_multiplier = 5.0
        align_frac = 0.5
        large_multiplier = 6.0
        MAX_TEXTS_PER_GROUP = 9
        MAX_CHARS_PER_GROUP = 1100
        text_text_threshold = threshold * 1.3

    elif layout_type == "full_page_single":
        # Few large elements: very aggressive grouping
        radius_multiplier = 8.0
        directional_multiplier = 12.0
        align_frac = 0.2
        large_multiplier = 15.0
        MAX_TEXTS_PER_GROUP = 20
        MAX_CHARS_PER_GROUP = 3000
        text_text_threshold = threshold * 2.0

    elif layout_type == "mixed_density":
        # Mixed areas: adaptive approach
        radius_multiplier = 2.5
        directional_multiplier = 4.0
        align_frac = 0.55
        large_multiplier = 5.0
        MAX_TEXTS_PER_GROUP = 8
        MAX_CHARS_PER_GROUP = 900
        text_text_threshold = threshold

    else:  # Default fallback
        radius_multiplier = 2.0
        directional_multiplier = 3.0
        align_frac = 0.6
        large_multiplier = 4.0
        MAX_TEXTS_PER_GROUP = 7
        MAX_CHARS_PER_GROUP = 800
        text_text_threshold = threshold

    # Collect nodes and maps for full objects: prefix with kind to avoid id clashes
    images = []
    texts = []
    images_map = {}
    texts_map = {}
    for im in page.get("images", []):
        bbox = im.get("bbox", [])
        if len(bbox) == 4:
            # Skip blocked images (too large or too small)
            if im.get("blocked"):
                if DEBUG_GROUPER:
                    print(
                        f"[group_blocks] Skipping blocked image {im.get('id')} (reason={im.get('blocked_reason')})"
                    )
                continue
            key = f"img:{im.get('id')}"
            images_map[key] = im
            images.append((key, bbox))
    for tb in page.get("texts", []):
        bbox = tb.get("bbox", [])
        if len(bbox) == 4:
            # Skip blocked text blocks (like headers/footers or odd font sizes)
            if tb.get("blocked"):
                if DEBUG_GROUPER:
                    print(
                        f"[group_blocks] Skipping blocked text {tb.get('id')} (reason={tb.get('blocked_reason')})"
                    )
                continue
            key = f"txt:{tb.get('id')}"
            texts_map[key] = tb
            texts.append((key, bbox))

    dsu = DSU()

    # thresholds & weights to avoid runaway group growth (layout-specific values set above)
    TEXT_COUNT_WEIGHT = 10.0
    CHAR_COUNT_WEIGHT = 0.01
    # If not earlier set by env, default to False
    try:
        if "DEBUG_GROUPER" not in locals():
            DEBUG_GROUPER = False
    except Exception:
        DEBUG_GROUPER = False

    # Step 1: group text-to-text using layout-specific threshold (preserve multi-line caption grouping)
    tn = len(texts)
    for i in range(tn):
        for j in range(i + 1, tn):
            a_key, a_bbox = texts[i]
            b_key, b_bbox = texts[j]
            try:
                if mode == "expand":
                    ax0, ay0, ax1, ay1 = a_bbox
                    bx0, by0, bx1, by1 = b_bbox
                    a_ex = (
                        ax0 - text_text_threshold,
                        ay0 - text_text_threshold,
                        ax1 + text_text_threshold,
                        ay1 + text_text_threshold,
                    )
                    b_ex = (
                        bx0 - text_text_threshold,
                        by0 - text_text_threshold,
                        bx1 + text_text_threshold,
                        by1 + text_text_threshold,
                    )
                    # Replace strict overlap condition with a distance threshold.
                    # This avoids missing group links when boxes don't strictly overlap
                    # but are near enough to be logically part of the same block.
                    dist = rect_distance(a_bbox, b_bbox)
                    if dist <= text_text_threshold:
                        dsu.union(a_key, b_key)
                else:
                    dist = rect_distance(a_bbox, b_bbox)
                    if dist <= text_text_threshold:
                        dsu.union(a_key, b_key)
            except Exception:
                dist = rect_distance(a_bbox, b_bbox)
                if dist <= text_text_threshold:
                    dsu.union(a_key, b_key)

    # Step 2: merge images that overlap (so nearby small slices become one anchor)
    in_ = len(images)
    for i in range(in_):
        for j in range(i + 1, in_):
            a_key, a_bbox = images[i]
            b_key, b_bbox = images[j]
            ax0, ay0, ax1, ay1 = a_bbox
            bx0, by0, bx1, by1 = b_bbox
            a_ex = (ax0 - threshold, ay0 - threshold, ax1 + threshold, ay1 + threshold)
            b_ex = (bx0 - threshold, by0 - threshold, bx1 + threshold, by1 + threshold)
            overlap = not (
                a_ex[2] < b_ex[0]
                or b_ex[2] < a_ex[0]
                or a_ex[3] < b_ex[1]
                or b_ex[3] < a_ex[1]
            )
            if overlap:
                dsu.union(a_key, b_key)

    # Step 3: Anchor texts to nearest/overlapping image
    # For each text (or text-component) prefer images whose expanded bbox overlaps;
    # otherwise pick nearest image within radius. If still none, try directional
    # heuristics (text to the right/below/left/top of image with small gap and
    # reasonable alignment) to attach captions placed beside or under images.
    # (radius_multiplier, directional_multiplier, align_frac set above based on layout)
    # Keep track of images that have already accepted a text in this pass
    used_img_keys = set()

    for t_key, t_bbox in texts:
        best_img = None
        best_dist = None
        tx0, ty0, tx1, ty1 = t_bbox
        tcx = (tx0 + tx1) / 2.0
        tcy = (ty0 + ty1) / 2.0
        overlapping_imgs = []
        for i_key, i_bbox in images:
            ix0, iy0, ix1, iy1 = i_bbox
            i_ex = (ix0 - threshold, iy0 - threshold, ix1 + threshold, iy1 + threshold)
            # overlap check between expanded image and text bbox
            if not (i_ex[2] < tx0 or tx1 < i_ex[0] or i_ex[3] < ty0 or ty1 < i_ex[1]):
                overlapping_imgs.append((i_key, i_bbox))
        if overlapping_imgs:
            # choose nearest overlapping image by rect distance
            for i_key, i_bbox in overlapping_imgs:
                d = rect_distance(i_bbox, t_bbox)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_img = i_key
        else:
            # no overlap; pick nearest image if within radius*threshold
            for i_key, i_bbox in images:
                d = rect_distance(i_bbox, t_bbox)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_img = i_key
            if best_dist is not None and best_dist > threshold * radius_multiplier:
                best_img = None

        # directional heuristics if no best yet
        # We build a direction priority list so that sidebar pages can prefer
        # horizontal attachments (text beside images) while other layouts can
        # keep the default vertical-first ordering.
        direction_priority = ["below", "right", "above", "left"]
        try:
            if page.get("_detected_layout") == "sidebar":
                # compute average image center to pick which horizontal side is the
                # likely caption side (if images are on left, captions are usually
                # to the right and vice-versa).
                page_w = page.get("width") or 0
                img_centers_x = []
                for im in page.get("images", []):
                    bb = im.get("bbox", [])
                    if len(bb) == 4:
                        img_centers_x.append((bb[0] + bb[2]) / 2.0)
                if img_centers_x and page_w:
                    avg_cx = sum(img_centers_x) / len(img_centers_x)
                    if avg_cx < (page_w / 2.0):
                        # images primarily on left -> prefer text to the right
                        direction_priority = ["right", "below", "above", "left"]
                    else:
                        # images primarily on right -> prefer text to the left
                        direction_priority = ["left", "below", "above", "right"]
                else:
                    # fallback to horizontal precedence
                    direction_priority = ["right", "below", "above", "left"]
        except Exception:
            direction_priority = ["below", "right", "above", "left"]

        if best_img is None:
            for dir_choice in direction_priority:
                if best_img is not None:
                    break
                for i_key, i_bbox in images:
                    ix0, iy0, ix1, iy1 = i_bbox
                    # below: text is below image
                    if dir_choice == "below":
                        if ty0 >= iy1:
                            gap = ty0 - iy1
                            horiz_overlap = max(0, min(tx1, ix1) - max(tx0, ix0))
                            img_w = ix1 - ix0 if ix1 > ix0 else 1.0
                            if gap <= threshold * directional_multiplier and (
                                horiz_overlap >= img_w * (align_frac / 2)
                                or abs(tcx - (ix0 + ix1) / 2.0) <= img_w * align_frac
                            ):
                                best_img = i_key
                                break
                    # right: text is to the right
                    elif dir_choice == "right":
                        if tx0 >= ix1:
                            gap = tx0 - ix1
                            vert_overlap = max(0, min(ty1, iy1) - max(ty0, iy0))
                            img_h = iy1 - iy0 if iy1 > iy0 else 1.0
                            if gap <= threshold * directional_multiplier and (
                                vert_overlap >= img_h * (align_frac / 2)
                                or abs(tcy - (iy0 + iy1) / 2.0) <= img_h * align_frac
                            ):
                                best_img = i_key
                                break
                    # above: text is above image
                    elif dir_choice == "above":
                        if ty1 <= iy0:
                            gap = iy0 - ty1
                            horiz_overlap = max(0, min(tx1, ix1) - max(tx0, ix0))
                            img_w = ix1 - ix0 if ix1 > ix0 else 1.0
                            if gap <= threshold * directional_multiplier and (
                                horiz_overlap >= img_w * (align_frac / 2)
                                or abs(tcx - (ix0 + ix1) / 2.0) <= img_w * align_frac
                            ):
                                best_img = i_key
                                break
                    # left: text is to the left
                    elif dir_choice == "left":
                        if tx1 <= ix0:
                            gap = ix0 - tx1
                            vert_overlap = max(0, min(ty1, iy1) - max(ty0, iy0))
                            img_h = iy1 - iy0 if iy1 > iy0 else 1.0
                            if gap <= threshold * directional_multiplier and (
                                vert_overlap >= img_h * (align_frac / 2)
                                or abs(tcy - (iy0 + iy1) / 2.0) <= img_h * align_frac
                            ):
                                best_img = i_key
                                break

        if best_img and best_img not in used_img_keys:
            # Attach this text block to the chosen image and mark the image used
            dsu.union(t_key, best_img)
            used_img_keys.add(best_img)

    # build components from all nodes (images+texts)
    comps = {}
    # include images
    for key, bbox in images:
        root = dsu.find(key)
        comps.setdefault(root, []).append((key, bbox, "image"))
    for key, bbox in texts:
        root = dsu.find(key)
        comps.setdefault(root, []).append((key, bbox, "text"))

    groups = []
    gid = 0
    for root, items in comps.items():
        gid += 1
        img_objs = []
        txt_objs = []
        xs = []
        ys = []
        x1s = []
        y1s = []
        for key, bbox, kind in items:
            x0, y0, x1, y1 = bbox
            xs.append(x0)
            ys.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            if kind == "image":
                # insert the full image block object
                obj = images_map.get(key)
                if obj is not None:
                    img_objs.append(obj)
            else:
                obj = texts_map.get(key)
                if obj is not None:
                    txt_objs.append(obj)
        group_bbox = [min(xs), min(ys), max(x1s), max(y1s)]
        groups.append(
            {
                "id": f"group_{gid}",
                "bbox": group_bbox,
                "images": img_objs,
                "texts": txt_objs,
            }
        )

    # Post-processing pass: try to attach remaining text-only groups to nearby
    # image-only groups using a larger search radius. This helps when captions
    # are slightly further away than the base threshold.
    # Build index lists
    text_only_idxs = []
    img_only_idxs = []
    for i, g in enumerate(groups):
        if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0:
            text_only_idxs.append(i)
        if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0:
            img_only_idxs.append(i)

    merged = set()
    large_multiplier = 4.0
    attach_epsilon = 0.10  # allow 10% margin when comparing distances

    def text_chars_in_group(g: dict) -> int:
        total = 0
        for t in g.get("texts", []):
            if isinstance(t, dict):
                txt = t.get("content") or t.get("text") or ""
            else:
                txt = str(t)
            if isinstance(txt, str):
                total += len(txt)
        return total

    def text_count_in_group(g: dict) -> int:
        return len(g.get("texts", []))

    # Ensure we only attach at most one text group to each image-group per iteration
    used_img_idxs = set()
    # process the smallest text groups first (prefer to attach small captions)
    text_only_order = sorted(
        text_only_idxs,
        key=lambda i: (text_count_in_group(groups[i]), text_chars_in_group(groups[i])),
    )
    for ti in text_only_order:
        if ti in merged:
            continue
        tgroup = groups[ti]
        best_j = None
        best_d = None
        best_chars = None
        best_score = None
        for ji in img_only_idxs:
            if ji in merged or ji in used_img_idxs:
                continue
            ig = groups[ji]
            # compute the union bbox if we attach ig into tgroup
            all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
            all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
            all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
            all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
            new_bbox = [all_x0, all_y0, all_x1, all_y1]
            # check that this new bbox would not overlap any other existing group
            would_overlap = False
            for k, other in enumerate(groups):
                if k == ti or k == ji or k in merged:
                    continue
                if rects_overlap(new_bbox, other["bbox"]):
                    would_overlap = True
                    break
            if would_overlap:
                # skip this candidate as it would create overlapping groups
                if DEBUG_GROUPER:
                    print(
                        f"[group_blocks] Skipping candidate img idx {ji} for text idx {ti} due to bbox overlap"
                    )
                continue
            d = rect_distance(tgroup["bbox"], ig["bbox"])
            # prefer image-groups that have fewer text blocks/chars, score by distance+penalties
            ch_count = text_count_in_group(ig)
            ch_chars = text_chars_in_group(ig)
            score = d + (TEXT_COUNT_WEIGHT * ch_count) + (CHAR_COUNT_WEIGHT * ch_chars)
            if best_d is None:
                best_d = d
                best_j = ji
                best_chars = ch_chars
                best_score = score
            else:
                if score < best_score:
                    best_d = d
                    best_j = ji
                    best_chars = ch_chars
                    best_score = score
                else:
                    # if within epsilon distance, prefer the group with fewer text chars
                    if best_d and d <= best_d * (1.0 + attach_epsilon):
                        ch = text_chars_in_group(ig)
                        sc = (
                            d
                            + (TEXT_COUNT_WEIGHT * text_count_in_group(ig))
                            + (CHAR_COUNT_WEIGHT * ch)
                        )
                        if sc < best_score:
                            best_chars = ch
                            best_j = ji
                            best_score = sc
        if (
            best_j is not None
            and best_d is not None
            and best_d <= threshold * large_multiplier
        ):
            # merge ji into ti (attach image to the text group)
            ig = groups[best_j]
            # don't attach into a text group that is already huge unless forced fallback
            if (
                text_count_in_group(tgroup) >= MAX_TEXTS_PER_GROUP
                and text_chars_in_group(tgroup) >= MAX_CHARS_PER_GROUP
            ):
                # skip attachment to this tgroup
                if DEBUG_GROUPER:
                    print(
                        f"[group_blocks] Skipping attach img->text due to large tgroup: tgroup id {tgroup.get('id')} count {text_count_in_group(tgroup)} chars {text_chars_in_group(tgroup)}"
                    )
                continue
            # append images from ig into tgroup
            tgroup["images"].extend(ig.get("images", []))
            # recompute bbox
            all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
            all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
            all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
            all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
            tgroup["bbox"] = [all_x0, all_y0, all_x1, all_y1]
            merged.add(best_j)
            used_img_idxs.add(best_j)

    # Remove merged groups (those that were absorbed)
    out_groups = []
    for i, g in enumerate(groups):
        if i in merged:
            continue
        out_groups.append(g)

    # Forced iterative matching: repeatedly try larger radii to match lone
    # image-only groups with lone text-only groups. We pick the closest text
    # for each lone image. Matching is restricted to the four cardinal
    # directions (left/right/top/bottom) to avoid diagonal matches. Once a
    # Repeat post-processing, forced iterative matching, and final pass until
    # no more attachments can be made or a safety iteration cap is reached.
    ITER_MAX = 12
    # large_multiplier set above based on layout
    attach_epsilon = 0.10  # allow 10% margin when comparing distances
    max_multiplier = 4096.0

    for iteration in range(ITER_MAX):
        made_any_attachment = False

        # Post-processing pass: try to attach remaining text-only groups to nearby
        # image-only groups using a larger search radius.
        text_only_idxs = []
        img_only_idxs = []
        for i, g in enumerate(groups):
            if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0:
                text_only_idxs.append(i)
            if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0:
                img_only_idxs.append(i)

        merged = set()

        # Ensure we only attach at most one text group to each image-group per iteration
        used_img_idxs = set()
        # process the smallest text groups first (prefer to attach small captions)
        text_only_order = sorted(
            text_only_idxs,
            key=lambda i: (
                text_count_in_group(groups[i]),
                text_chars_in_group(groups[i]),
            ),
        )
        for ti in text_only_order:
            if ti in merged:
                continue
            tgroup = groups[ti]
            best_j = None
            best_d = None
            best_chars = None
            best_score = None
            for ji in img_only_idxs:
                if ji in merged or ji in used_img_idxs:
                    continue
                ig = groups[ji]
                # compute the union bbox if we attach ig into tgroup
                all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
                all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
                all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
                all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
                new_bbox = [all_x0, all_y0, all_x1, all_y1]
                # check that this new bbox would not overlap any other existing group
                would_overlap = False
                for k, other in enumerate(groups):
                    if k == ti or k == ji or k in merged:
                        continue
                    if rects_overlap(new_bbox, other["bbox"]):
                        would_overlap = True
                        break
                if would_overlap:
                    # skip this candidate as it would create overlapping groups
                    if DEBUG_GROUPER:
                        print(
                            f"[group_blocks] Skipping candidate img idx {ji} for text idx {ti} due to bbox overlap"
                        )
                    continue
                d = rect_distance(tgroup["bbox"], ig["bbox"])
                # prefer image-groups that have fewer text blocks/chars, score by distance+penalties
                ch_count = text_count_in_group(ig)
                ch_chars = text_chars_in_group(ig)
                score = (
                    d + (TEXT_COUNT_WEIGHT * ch_count) + (CHAR_COUNT_WEIGHT * ch_chars)
                )
                if best_d is None:
                    best_d = d
                    best_j = ji
                    best_chars = ch_chars
                    best_score = score
                else:
                    if score < best_score:
                        best_d = d
                        best_j = ji
                        best_chars = ch_chars
                        best_score = score
                    else:
                        # if within epsilon distance, prefer the group with fewer text chars
                        if best_d and d <= best_d * (1.0 + attach_epsilon):
                            ch = text_chars_in_group(ig)
                            sc = (
                                d
                                + (TEXT_COUNT_WEIGHT * text_count_in_group(ig))
                                + (CHAR_COUNT_WEIGHT * ch)
                            )
                            if sc < best_score:
                                best_chars = ch
                                best_j = ji
                                best_score = sc
            if (
                best_j is not None
                and best_d is not None
                and best_d <= threshold * large_multiplier
            ):
                # merge ji into ti (attach image to the text group)
                ig = groups[best_j]
                # don't attach into a text group that is already huge unless forced fallback
                if (
                    text_count_in_group(tgroup) >= MAX_TEXTS_PER_GROUP
                    and text_chars_in_group(tgroup) >= MAX_CHARS_PER_GROUP
                ):
                    # skip attachment to this tgroup
                    if DEBUG_GROUPER:
                        print(
                            f"[group_blocks] Skipping attach img->text due to large tgroup: tgroup id {tgroup.get('id')} count {text_count_in_group(tgroup)} chars {text_chars_in_group(tgroup)}"
                        )
                    continue
                # append images from ig into tgroup
                tgroup["images"].extend(ig.get("images", []))
                # recompute bbox
                all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
                all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
                all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
                all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
                tgroup["bbox"] = [all_x0, all_y0, all_x1, all_y1]
                merged.add(best_j)
                used_img_idxs.add(best_j)
                made_any_attachment = True

        # Remove merged groups (those that were absorbed)
        out_groups = []
        for i, g in enumerate(groups):
            if i in merged:
                continue
            out_groups.append(g)

        # Forced iterative matching: repeatedly try larger radii to match lone
        # image-only groups with lone text-only groups. We pick the closest text
        # for each lone image. Matching is restricted to the four cardinal
        # directions (left/right/top/bottom) to avoid diagonal matches. Once a
        # text group is attached to an image group, both are removed from the
        # matching queue and will not be considered again.
        groups_work = out_groups
        start_mult = large_multiplier
        # Each pass attaches at most 1 text to each lone image.
        while True:
            lone_img_idxs = [
                i
                for i, g in enumerate(groups_work)
                if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0
            ]
            lone_txt_idxs = [
                i
                for i, g in enumerate(groups_work)
                if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
            ]
            if not lone_img_idxs or not lone_txt_idxs:
                break

            id_to_idx = {g["id"]: i for i, g in enumerate(groups_work)}
            attach_pairs = []  # (img_id, txt_id)
            used_text_ids = set()
            used_img_ids = set()

            # process images with fewer text chars first (so small groups get preference)
            lone_img_idxs_sorted = sorted(
                lone_img_idxs, key=lambda i: text_chars_in_group(groups_work[i])
            )
            for ii in lone_img_idxs_sorted:
                img_g = groups_work[ii]
                img_id = img_g["id"]
                if img_id in used_img_ids:
                    continue
                # if image group is already large, skip until next pass
                if (
                    text_count_in_group(img_g) >= MAX_TEXTS_PER_GROUP
                    or text_chars_in_group(img_g) >= MAX_CHARS_PER_GROUP
                ):
                    if DEBUG_GROUPER:
                        print(
                            f"[group_blocks] Skipping image {img_id} in forced pass: already large (texts={text_count_in_group(img_g)} chars={text_chars_in_group(img_g)})"
                        )
                    continue
                best_txt_id = None
                best_d = None
                mult = start_mult
                while mult <= max_multiplier and best_txt_id is None:
                    for j in lone_txt_idxs:
                        txt_g = groups_work[j]
                        txt_id = txt_g["id"]
                        if txt_id in used_text_ids:
                            continue
                        tx0, ty0, tx1, ty1 = txt_g["bbox"]
                        tcx = (tx0 + tx1) / 2.0
                        tcy = (ty0 + ty1) / 2.0
                        ix0, iy0, ix1, iy1 = img_g["bbox"]
                        icx = (ix0 + ix1) / 2.0
                        icy = (iy0 + iy1) / 2.0
                        dx = tcx - icx
                        dy = tcy - icy
                        # prefer cardinally aligned candidates
                        if abs(dx) >= abs(dy):
                            vert_overlap = max(0, min(iy1, ty1) - max(iy0, ty0))
                            img_h = max(1.0, iy1 - iy0)
                            if (
                                vert_overlap >= img_h * (align_frac / 2)
                                or abs(tcy - icy) <= img_h * align_frac
                            ):
                                d = rect_distance(img_g["bbox"], txt_g["bbox"])
                                if d <= threshold * mult:
                                    ch = text_chars_in_group(txt_g)
                                    # pick the smallest text char group in ties
                                    if best_d is None or d < best_d:
                                        best_d = d
                                        best_txt_id = txt_id
                                        best_chars = ch
                                    else:
                                        if d <= best_d * (
                                            1.0 + attach_epsilon
                                        ) and ch < (best_chars or 0):
                                            best_d = d
                                            best_txt_id = txt_id
                                            best_chars = ch
                        else:
                            horiz_overlap = max(0, min(ix1, tx1) - max(ix0, tx0))
                            img_w = max(1.0, ix1 - ix0)
                            if (
                                horiz_overlap >= img_w * (align_frac / 2)
                                or abs(tcx - icx) <= img_w * align_frac
                            ):
                                d = rect_distance(img_g["bbox"], txt_g["bbox"])
                                if d <= threshold * mult:
                                    ch = text_chars_in_group(txt_g)
                                    if best_d is None or d < best_d:
                                        best_d = d
                                        best_txt_id = txt_id
                                        best_chars = ch
                                    else:
                                        if d <= best_d * (
                                            1.0 + attach_epsilon
                                        ) and ch < (best_chars or 0):
                                            best_d = d
                                            best_txt_id = txt_id
                                            best_chars = ch
                        # when distances are similar, prefer text groups with fewer chars
                    if best_txt_id is None:
                        mult *= 2.0
                if best_txt_id is not None:
                    # check whether this would overflow the image group limits
                    txt_obj = next(
                        (g for g in groups_work if g["id"] == best_txt_id), None
                    )
                    if txt_obj:
                        img_curr_count = text_count_in_group(img_g)
                        img_curr_chars = text_chars_in_group(img_g)
                        add_count = text_count_in_group(txt_obj)
                        add_chars = text_chars_in_group(txt_obj)
                        if (
                            img_curr_count + add_count <= MAX_TEXTS_PER_GROUP
                            and img_curr_chars + add_chars <= MAX_CHARS_PER_GROUP
                        ):
                            # ensure union bbox won't overlap other existing groups in this work set
                            all_x0 = min(img_g["bbox"][0], txt_obj["bbox"][0])
                            all_y0 = min(img_g["bbox"][1], txt_obj["bbox"][1])
                            all_x1 = max(img_g["bbox"][2], txt_obj["bbox"][2])
                            all_y1 = max(img_g["bbox"][3], txt_obj["bbox"][3])
                            new_bbox = [all_x0, all_y0, all_x1, all_y1]
                            would_overlap = False
                            for k, other in enumerate(groups_work):
                                if other["id"] == img_id or other["id"] == best_txt_id:
                                    continue
                                if rects_overlap(new_bbox, other["bbox"]):
                                    would_overlap = True
                                    break
                            if would_overlap:
                                if DEBUG_GROUPER:
                                    print(
                                        f"[group_blocks] Skipping attach txt {best_txt_id} -> img {img_id} in forced pass due to would-overlap"
                                    )
                            else:
                                attach_pairs.append((img_id, best_txt_id))
                                made_any_attachment = True
                        else:
                            # skip since this would make image group too large; try a different image later
                            if DEBUG_GROUPER:
                                print(
                                    f"[group_blocks] Not attaching txt {best_txt_id} to img {img_id} because it would overflow (img_count={img_curr_count} + add={add_count}, chars={img_curr_chars}+{add_chars})"
                                )
                            pass
                    else:
                        attach_pairs.append((img_id, best_txt_id))
                        made_any_attachment = True
                    used_text_ids.add(best_txt_id)
                    used_img_ids.add(img_id)

            if not attach_pairs:
                break

            # perform attachments and remove text groups
            remove_ids = set()
            for img_id, txt_id in attach_pairs:
                img_obj = next((g for g in groups_work if g["id"] == img_id), None)
                txt_obj = next((g for g in groups_work if g["id"] == txt_id), None)
                if not img_obj or not txt_obj:
                    continue
                # if img group already large, skip this attachment (should not happen due to earlier checks)
                if (
                    text_count_in_group(img_obj) >= MAX_TEXTS_PER_GROUP
                    and text_chars_in_group(img_obj) >= MAX_CHARS_PER_GROUP
                ):
                    continue
                img_obj["texts"].extend(txt_obj.get("texts", []))
                all_x0 = min(img_obj["bbox"][0], txt_obj["bbox"][0])
                all_y0 = min(img_obj["bbox"][1], txt_obj["bbox"][1])
                all_x1 = max(img_obj["bbox"][2], txt_obj["bbox"][2])
                all_y1 = max(img_obj["bbox"][3], txt_obj["bbox"][3])
                img_obj["bbox"] = [all_x0, all_y0, all_x1, all_y1]
                remove_ids.add(txt_id)
            # remove text groups by id in descending order of index
            if remove_ids:
                id_to_idx = {g["id"]: i for i, g in enumerate(groups_work)}
                rm_idxs = sorted(
                    [id_to_idx[r] for r in remove_ids if r in id_to_idx], reverse=True
                )
                for idx in rm_idxs:
                    groups_work.pop(idx)

        # Final pass: attach any remaining lone text-only groups to the nearest
        # existing group that has images. This removes leftover lone text groups
        # by merging them into the closest image-containing group.
        final_mult = max_multiplier * 2.0
        text_only_idxs = [
            i
            for i, g in enumerate(groups_work)
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        ]
        candidate_idxs = [
            i for i, g in enumerate(groups_work) if len(g.get("images", [])) > 0
        ]
        to_remove = set()
        for ti in text_only_idxs:
            if ti in to_remove:
                continue
            best_j = None
            best_d = None
            best_chars = None
            best_score = None
            for cj in candidate_idxs:
                if cj in to_remove or cj == ti:
                    continue
                d = rect_distance(groups_work[ti]["bbox"], groups_work[cj]["bbox"])
                ch_count = text_count_in_group(groups_work[cj])
                ch = text_chars_in_group(groups_work[cj])
                score = d + (TEXT_COUNT_WEIGHT * ch_count) + (CHAR_COUNT_WEIGHT * ch)
                preferred = ch_count < MAX_TEXTS_PER_GROUP and ch < MAX_CHARS_PER_GROUP
                if best_score is None:
                    best_score = score
                    best_j = cj
                    best_d = d
                    best_chars = ch
                    best_preferred = preferred
                else:
                    if preferred and not best_preferred:
                        best_score = score
                        best_j = cj
                        best_d = d
                        best_chars = ch
                        best_preferred = preferred
                    elif preferred == best_preferred and score < best_score:
                        best_score = score
                        best_j = cj
                        best_d = d
                        best_chars = ch
            if (
                best_j is not None
                and best_d is not None
                and best_d <= threshold * final_mult
            ):
                # attach text group into the candidate group
                tgt = groups_work[best_j]
                src = groups_work[ti]
                # if target already has many texts/chars, skip unless forced fallback
                if (
                    text_count_in_group(tgt) >= MAX_TEXTS_PER_GROUP
                    and text_chars_in_group(tgt) >= MAX_CHARS_PER_GROUP
                ):
                    continue
                # compute union bbox and ensure it won't overlap other groups
                all_x0 = min(tgt["bbox"][0], src["bbox"][0])
                all_y0 = min(tgt["bbox"][1], src["bbox"][1])
                all_x1 = max(tgt["bbox"][2], src["bbox"][2])
                all_y1 = max(tgt["bbox"][3], src["bbox"][3])
                new_bbox = [all_x0, all_y0, all_x1, all_y1]
                would_overlap = False
                for k, other in enumerate(groups_work):
                    if k == best_j or k == ti:
                        continue
                    if rects_overlap(new_bbox, other["bbox"]):
                        would_overlap = True
                        break
                if would_overlap:
                    if DEBUG_GROUPER:
                        print(
                            f"[group_blocks] Skipping final attach txt idx {ti} -> tgt idx {best_j} due to bbox overlap"
                        )
                    continue
                tgt["texts"].extend(src.get("texts", []))
                # expand bbox
                tgt["bbox"] = new_bbox
                to_remove.add(ti)
                made_any_attachment = True

        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                if 0 <= idx < len(groups_work):
                    groups_work.pop(idx)

        # prepare for next iteration: set groups to the current workset
        groups = groups_work

        if not made_any_attachment:
            break

    return groups


def attach_remaining_groups(
    groups: List[dict], threshold: float = 5.0, page: dict = None
) -> tuple:
    """Greedily attach remaining lone text-only groups into image-containing
    groups when safe (no bbox overlap, within capacity and distance limits).

    Returns (new_groups, attachments_made).
    """
    MAX_TEXTS_PER_GROUP = 7
    MAX_CHARS_PER_GROUP = 800
    TEXT_COUNT_WEIGHT = 10.0
    CHAR_COUNT_WEIGHT = 0.01
    FINAL_MULT = 8.0

    def text_chars(g: dict) -> int:
        total = 0
        for t in g.get("texts", []):
            if isinstance(t, dict):
                s = t.get("content") or t.get("text") or ""
            else:
                s = str(t)
            if isinstance(s, str):
                total += len(s)
        return total

    def text_count(g: dict) -> int:
        return len(g.get("texts", []))

    attachments = 0
    # Alignment fraction used in directional heuristics (fraction of width/height for overlap checks)
    align_frac = 0.6
    groups_work = groups

    # helper: merge whole groups that overlap by a large fraction
    def _bbox_area(b):
        try:
            x0, y0, x1, y1 = b
            return max(0.0, x1 - x0) * max(0.0, y1 - y0)
        except Exception:
            return 0.0

    def _intersection_area(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def _merge_whole_groups(groups_list: List[dict], overlap_frac: float = 0.2):
        """Iteratively merge entire groups when intersection covers >= overlap_frac
        of the smaller group's area. Returns new group list.
        """
        if not groups_list:
            return groups_list
        n = len(groups_list)
        changed = True
        while changed:
            changed = False
            removed = [False] * n
            new_groups = []
            for i in range(n):
                if removed[i]:
                    continue
                gi = groups_list[i]
                for j in range(i + 1, n):
                    if removed[j]:
                        continue
                    gj = groups_list[j]
                    ia = _intersection_area(
                        gi.get("bbox", [0, 0, 0, 0]), gj.get("bbox", [0, 0, 0, 0])
                    )
                    if ia <= 0.0:
                        continue
                    a_area = _bbox_area(gi.get("bbox", [0, 0, 0, 0]))
                    b_area = _bbox_area(gj.get("bbox", [0, 0, 0, 0]))
                    smallest = min(a_area or float("inf"), b_area or float("inf"))
                    if smallest == 0.0 or smallest == float("inf"):
                        continue
                    if ia / smallest >= overlap_frac:
                        # merge whole gj into gi
                        gi["images"].extend(gj.get("images", []))
                        gi["texts"].extend(gj.get("texts", []))
                        gi["bbox"] = [
                            min(gi["bbox"][0], gj["bbox"][0]),
                            min(gi["bbox"][1], gj["bbox"][1]),
                            max(gi["bbox"][2], gj["bbox"][2]),
                            max(gi["bbox"][3], gj["bbox"][3]),
                        ]
                        removed[j] = True
                        changed = True
                new_groups.append(gi)
            # rebuild groups_list from new_groups keeping non-removed
            groups_list = [g for idx, g in enumerate(new_groups) if not removed[idx]]
            n = len(groups_list)
        return groups_list

    while True:
        made = False
        text_idxs = [
            i
            for i, g in enumerate(groups_work)
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        ]
        cand_idxs = [
            i for i, g in enumerate(groups_work) if len(g.get("images", [])) > 0
        ]
        if not text_idxs or not cand_idxs:
            break

        text_order = sorted(
            text_idxs,
            key=lambda i: (text_count(groups_work[i]), text_chars(groups_work[i])),
        )
        removed = set()
        for ti in text_order:
            if ti in removed:
                continue
            src = groups_work[ti]
            best = None
            best_score = None
            for cj in cand_idxs:
                if cj == ti:
                    continue
                tgt = groups_work[cj]
                # capacity check
                if text_count(tgt) + text_count(src) > MAX_TEXTS_PER_GROUP:
                    continue
                if text_chars(tgt) + text_chars(src) > MAX_CHARS_PER_GROUP:
                    continue
                # union bbox and check overlap with others
                all_x0 = min(tgt["bbox"][0], src["bbox"][0])
                all_y0 = min(tgt["bbox"][1], src["bbox"][1])
                all_x1 = max(tgt["bbox"][2], src["bbox"][2])
                all_y1 = max(tgt["bbox"][3], src["bbox"][3])
                new_bbox = [all_x0, all_y0, all_x1, all_y1]
                overlaps = False
                for k, other in enumerate(groups_work):
                    if k == ti or k == cj:
                        continue
                    if rects_overlap(new_bbox, other["bbox"]):
                        overlaps = True
                        break
                if overlaps:
                    continue
                d = rect_distance(tgt["bbox"], src["bbox"])
                if d > threshold * FINAL_MULT:
                    continue
                score = (
                    d
                    + TEXT_COUNT_WEIGHT * text_count(tgt)
                    + CHAR_COUNT_WEIGHT * text_chars(tgt)
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = cj
            if best is not None:
                tgt = groups_work[best]
                tgt["texts"].extend(src.get("texts", []))
                tgt["bbox"] = [
                    min(tgt["bbox"][0], src["bbox"][0]),
                    min(tgt["bbox"][1], src["bbox"][1]),
                    max(tgt["bbox"][2], src["bbox"][2]),
                    max(tgt["bbox"][3], src["bbox"][3]),
                ]
                removed.add(ti)
                made = True
                attachments += 1
        if not made:
            break
        for idx in sorted(removed, reverse=True):
            if 0 <= idx < len(groups_work):
                groups_work.pop(idx)

    # FINAL FALLBACK: If there are remaining lone text groups that could not
    # be safely attached (due to overlap/capacity), attempt a forced attach pass
    # where we attach each remaining text-only group to its nearest image
    # containing group. This is a last-resort to avoid leaving any lone text
    # groups at the end of processing.
    # NOTE: This fallback may increase incorrect attachments in rare layouts,
    # but it satisfies the requirement that no text block remains ungrouped.
    remaining_text_idxs = [
        i
        for i, g in enumerate(groups_work)
        if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
    ]
    img_idxs = [i for i, g in enumerate(groups_work) if len(g.get("images", [])) > 0]
    # If there are no existing image-groups but the original page contained
    # images that were blocked (for example filtered out earlier), allow the
    # final-resolution stage to consider blocked images as attachment targets.
    # This is an optional behaviour (requires the caller to pass `page`).
    if not img_idxs and page is not None:
        blocked_images = [im for im in page.get("images", []) if im.get("blocked")]
        if blocked_images:
            # append blocked image groups as candidate groups (mark them so we
            # can identify later and keep traceability)
            base_len = len(groups_work)
            for im in blocked_images:
                groups_work.append(
                    {
                        "id": f"blocked_img_group_{im.get('id')}",
                        "bbox": im.get("bbox", []),
                        "images": [im],
                        "texts": [],
                        "_blocked_image_anchor": True,
                    }
                )
            img_idxs = [
                i for i, g in enumerate(groups_work) if len(g.get("images", [])) > 0
            ]
    # First try a directional reverse-search from each text-only group: look for
    # a nearby image-containing group using the same direction priority used
    # earlier (bottom -> right -> top -> left). This tries to attach texts to
    # images based on cardinal relationships before falling back to nearest.
    directional_multiplier = 6.0
    if remaining_text_idxs and img_idxs:
        # attempt directional attachments repeatedly until no progress
        made_dir_attach = True
        while made_dir_attach:
            made_dir_attach = False
            remaining_text_idxs = [
                i
                for i, g in enumerate(groups_work)
                if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
            ]
            if not remaining_text_idxs:
                break
            for ti in list(remaining_text_idxs):
                tgroup = groups_work[ti]
                tx0, ty0, tx1, ty1 = tgroup["bbox"]
                tcx = (tx0 + tx1) / 2.0
                tcy = (ty0 + ty1) / 2.0
                attached = False

                # direction order: choose priority based on detected layout. For
                # sidebar layouts prefer horizontal attachments first (text beside
                # images). Otherwise use a sensible default ordering that favors
                # captions below images.
                try:
                    if page is not None and page.get("_detected_layout") == "sidebar":
                        # determine whether images are mostly on left or right
                        page_w = page.get("width") or 0
                        img_centers_x = []
                        for im in page.get("images", []):
                            bb = im.get("bbox", [])
                            if len(bb) == 4:
                                img_centers_x.append((bb[0] + bb[2]) / 2.0)
                        if img_centers_x and page_w:
                            avg_cx = sum(img_centers_x) / len(img_centers_x)
                            if avg_cx < (page_w / 2.0):
                                # images on left => prefer images left-of-text (i.e. attach text to right)
                                directions = [
                                    "left_of_text",
                                    "below_image",
                                    "above_image",
                                    "right_of_text",
                                ]
                            else:
                                # images on right => prefer images right-of-text (attach text to left)
                                directions = [
                                    "right_of_text",
                                    "below_image",
                                    "above_image",
                                    "left_of_text",
                                ]
                        else:
                            directions = [
                                "left_of_text",
                                "below_image",
                                "above_image",
                                "right_of_text",
                            ]
                    else:
                        # default ordering: below -> left -> above -> right
                        directions = [
                            "below_image",
                            "left_of_text",
                            "above_image",
                            "right_of_text",
                        ]
                except Exception:
                    directions = [
                        "below_image",
                        "left_of_text",
                        "above_image",
                        "right_of_text",
                    ]
                for dirname in directions:
                    if attached:
                        break
                    # scan candidate image groups
                    for ji in img_idxs:
                        tgt = groups_work[ji]
                        ix0, iy0, ix1, iy1 = tgt["bbox"]
                        # below_image: text is below image (ty0 >= iy1)
                        if dirname == "below_image":
                            if ty0 >= iy1:
                                gap = ty0 - iy1
                                horiz_overlap = max(0, min(tx1, ix1) - max(tx0, ix0))
                                img_w = ix1 - ix0 if ix1 > ix0 else 1.0
                                if gap <= threshold * directional_multiplier and (
                                    horiz_overlap >= img_w * (align_frac / 2)
                                    or abs(tcx - (ix0 + ix1) / 2.0)
                                    <= img_w * align_frac
                                ):
                                    # attach
                                    tgt["texts"].extend(tgroup.get("texts", []))
                                    tgt["bbox"] = [
                                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                                    ]
                                    tgroup["texts"] = []
                                    attachments += 1
                                    made_dir_attach = True
                                    attached = True
                                    break
                        elif dirname == "left_of_text":
                            # image is to left of text (ix1 <= tx0)
                            if ix1 <= tx0:
                                gap = tx0 - ix1
                                vert_overlap = max(0, min(ty1, iy1) - max(ty0, iy0))
                                img_h = iy1 - iy0 if iy1 > iy0 else 1.0
                                if gap <= threshold * directional_multiplier and (
                                    vert_overlap >= img_h * (align_frac / 2)
                                    or abs(tcy - (iy0 + iy1) / 2.0)
                                    <= img_h * align_frac
                                ):
                                    tgt["texts"].extend(tgroup.get("texts", []))
                                    tgt["bbox"] = [
                                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                                    ]
                                    tgroup["texts"] = []
                                    attachments += 1
                                    made_dir_attach = True
                                    attached = True
                                    break
                        elif dirname == "above_image":
                            # image is below text (iy0 >= ty1)
                            if iy0 >= ty1:
                                gap = iy0 - ty1
                                horiz_overlap = max(0, min(tx1, ix1) - max(tx0, ix0))
                                img_w = ix1 - ix0 if ix1 > ix0 else 1.0
                                if gap <= threshold * directional_multiplier and (
                                    horiz_overlap >= img_w * (align_frac / 2)
                                    or abs(tcx - (ix0 + ix1) / 2.0)
                                    <= img_w * align_frac
                                ):
                                    tgt["texts"].extend(tgroup.get("texts", []))
                                    tgt["bbox"] = [
                                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                                    ]
                                    tgroup["texts"] = []
                                    attachments += 1
                                    made_dir_attach = True
                                    attached = True
                                    break
                        elif dirname == "right_of_text":
                            # image is to right of text (ix0 >= tx1)
                            if ix0 >= tx1:
                                gap = ix0 - tx1
                                vert_overlap = max(0, min(ty1, iy1) - max(ty0, iy0))
                                img_h = iy1 - iy0 if iy1 > iy0 else 1.0
                                if gap <= threshold * directional_multiplier and (
                                    vert_overlap >= img_h * (align_frac / 2)
                                    or abs(tcy - (iy0 + iy1) / 2.0)
                                    <= img_h * align_frac
                                ):
                                    tgt["texts"].extend(tgroup.get("texts", []))
                                    tgt["bbox"] = [
                                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                                    ]
                                    tgroup["texts"] = []
                                    attachments += 1
                                    made_dir_attach = True
                                    attached = True
                                    break
                # end directions loop
            # end for ti

        # recompute remaining_text_idxs after directional attempts
        remaining_text_idxs = [
            i
            for i, g in enumerate(groups_work)
            if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
        ]
        for ti in remaining_text_idxs:
            # pick nearest image group by bbox distance
            best_j = None
            best_d = None
            tgroup = groups_work[ti]
            for ji in img_idxs:
                tgt = groups_work[ji]
                d = rect_distance(tgroup["bbox"], tgt["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = ji
            if best_j is not None:
                tgt = groups_work[best_j]
                # attach texts to target even if it causes bbox overlap
                tgt["texts"].extend(tgroup.get("texts", []))
                # expand bbox
                all_x0 = min(tgt["bbox"][0], tgroup["bbox"][0])
                all_y0 = min(tgt["bbox"][1], tgroup["bbox"][1])
                all_x1 = max(tgt["bbox"][2], tgroup["bbox"][2])
                all_y1 = max(tgt["bbox"][3], tgroup["bbox"][3])
                tgt["bbox"] = [all_x0, all_y0, all_x1, all_y1]
                attachments += 1
                # mark the text group as removed by setting its texts to empty
                tgroup["texts"] = []

        # finally remove any groups that became empty (text-only groups removed)
        groups_clean = [
            g
            for g in groups_work
            if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
        ]
        rem = [
            i
            for i, g in enumerate(groups_clean)
            if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
        ]
        if rem:
            all_idxs = [i for i, _ in enumerate(groups_clean) if True]
            for ti in list(rem):
                tgroup = groups_clean[ti]
                best_j = None
                best_d = None
                for ji in all_idxs:
                    if ji == ti:
                        continue
                    tgt = groups_clean[ji]
                    d = rect_distance(tgroup["bbox"], tgt["bbox"])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = ji
                if best_j is not None:
                    tgt = groups_clean[best_j]
                    tgt["texts"].extend(tgroup.get("texts", []))
                    tgt["bbox"] = [
                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                    ]
                    tgroup["texts"] = []
                    attachments += 1

            groups_clean = [
                g
                for g in groups_clean
                if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
            ]

        # If we reach here either directional/img attempts ran (and possibly
        # modified groups_work) or there were no image-groups. Make a FINAL
        # pass to merge any remaining text-only groups into nearest group
        # (including other text-only groups). This ensures there are no lone
        # text-only groups remaining.
        groups_clean = [
            g
            for g in groups_work
            if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
        ]
        rem2 = [
            i
            for i, g in enumerate(groups_clean)
            if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
        ]
        if rem2:
            all_idxs = [i for i, _ in enumerate(groups_clean) if True]
            for ti in list(rem2):
                tgroup = groups_clean[ti]
                best_j = None
                best_d = None
                for ji in all_idxs:
                    if ji == ti:
                        continue
                    tgt = groups_clean[ji]
                    d = rect_distance(tgroup["bbox"], tgt["bbox"])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = ji
                if best_j is not None:
                    tgt = groups_clean[best_j]
                    tgt["texts"].extend(tgroup.get("texts", []))
                    tgt["bbox"] = [
                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                    ]
                    tgroup["texts"] = []
                    attachments += 1

            groups_clean = [
                g
                for g in groups_clean
                if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
            ]

        # As a final cleanup step: attach any remaining image-only groups into
        # the nearest group (prefer groups that already contain text). This
        # avoids leaving lone image groups when there are text groups to attach
        # to — the user requested lone images be merged into their relative
        # groups.
        # Identify remaining image-only groups
        img_only_idxs = [
            i
            for i, g in enumerate(groups_clean)
            if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0
        ]
        if img_only_idxs:
            all_idxs = [i for i, _ in enumerate(groups_clean)]

            # process smaller image-only groups first (by bbox area)
            def area(b):
                try:
                    x0, y0, x1, y1 = b
                    return max(0, x1 - x0) * max(0, y1 - y0)
                except Exception:
                    return float("inf")

            img_order = sorted(
                img_only_idxs, key=lambda i: area(groups_clean[i]["bbox"])
            )
            for ii in list(img_order):
                if len(groups_clean[ii].get("images", [])) == 0:
                    continue
                src = groups_clean[ii]
                # prefer targets that have text content
                candidate_idxs = [
                    j
                    for j in all_idxs
                    if j != ii and len(groups_clean[j].get("texts", [])) > 0
                ]
                # fallback to any group
                if not candidate_idxs:
                    candidate_idxs = [j for j in all_idxs if j != ii]

                best_j = None
                best_d = None
                for cj in candidate_idxs:
                    tgt = groups_clean[cj]
                    d = rect_distance(src["bbox"], tgt["bbox"])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = cj

                if best_j is not None:
                    tgt = groups_clean[best_j]
                    # mark forced image attachments for auditing
                    tgt["_forced_image_attachments"] = (
                        tgt.get("_forced_image_attachments", 0) + 1
                    )
                    tgt["images"].extend(src.get("images", []))
                    tgt["bbox"] = [
                        min(tgt["bbox"][0], src["bbox"][0]),
                        min(tgt["bbox"][1], src["bbox"][1]),
                        max(tgt["bbox"][2], src["bbox"][2]),
                        max(tgt["bbox"][3], src["bbox"][3]),
                    ]
                    # remove images from source group so it becomes empty and will be cleaned
                    src["images"] = []

            # remove any groups that became empty (no images and no texts)
            groups_clean = [
                g
                for g in groups_clean
                if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
            ]

        try:
            OVERLAP_FRAC = float(os.getenv("PDF_GROUP_MERGE_OVERLAP_FRAC", "0.2"))
        except Exception:
            OVERLAP_FRAC = 0.2
        groups_clean = _merge_whole_groups(groups_clean, overlap_frac=OVERLAP_FRAC)
        return groups_clean, attachments

    # If directional/img fallback didn't run or didn't clear everything, do a
    # final unconditional pass to merge any remaining text-only groups into
    # their nearest neighbor group (any kind). This covers pages with no
    # images or other edge-cases so no text blocks remain lone.
    groups_clean = [
        g
        for g in groups_work
        if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
    ]
    rem_final = [
        i
        for i, g in enumerate(groups_clean)
        if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
    ]
    if rem_final:
        # Greedy iterative merge: continue running nearest-neighbour merges until
        # no text-only groups remain. This avoids ping-pong merges (A->B then B->A)
        # by removing empty groups between iterations and recomputing indices.
        while True:
            groups_clean = [
                g
                for g in groups_work
                if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
            ]
            rem_final = [
                i
                for i, g in enumerate(groups_clean)
                if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0
            ]
            if not rem_final:
                break

            all_idxs = [i for i, _ in enumerate(groups_clean)]
            made_any = False
            # Process the smallest text groups first to prefer attaching small
            # caption-like blocks into larger neighbours.
            rem_final_order = sorted(
                rem_final, key=lambda i: len(groups_clean[i].get("texts", []))
            )
            for ti in list(rem_final_order):
                # If the group has already been emptied by a previous attach, skip
                if len(groups_clean[ti].get("texts", [])) == 0:
                    continue
                tgroup = groups_clean[ti]
                best_j = None
                best_d = None
                for ji in all_idxs:
                    if ji == ti:
                        continue
                    tgt = groups_clean[ji]
                    d = rect_distance(tgroup["bbox"], tgt["bbox"])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = ji
                if best_j is not None:
                    tgt = groups_clean[best_j]
                    # mark forced attachments for auditing
                    tgt["_forced_attachments"] = tgt.get("_forced_attachments", 0) + 1
                    # move texts over to the chosen target
                    tgt["texts"].extend(tgroup.get("texts", []))
                    tgt["bbox"] = [
                        min(tgt["bbox"][0], tgroup["bbox"][0]),
                        min(tgt["bbox"][1], tgroup["bbox"][1]),
                        max(tgt["bbox"][2], tgroup["bbox"][2]),
                        max(tgt["bbox"][3], tgroup["bbox"][3]),
                    ]
                    # clear source texts as they've been absorbed
                    tgroup["texts"] = []
                    attachments += 1
                    made_any = True

            # Remove groups that became empty in this pass, then repeat until
            # rem_final is empty (i.e. no text-only groups remain).
            groups_work = [
                g
                for g in groups_clean
                if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
            ]
            if not made_any:
                # if no progress, break out to avoid infinite loop
                break

        # Final cleanup and return
        groups_clean = [
            g
            for g in groups_work
            if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
        ]
        try:
            OVERLAP_FRAC = float(os.getenv("PDF_GROUP_MERGE_OVERLAP_FRAC", "0.2"))
        except Exception:
            OVERLAP_FRAC = 0.2
        groups_clean = _merge_whole_groups(groups_clean, overlap_frac=OVERLAP_FRAC)
        return groups_clean, attachments

    # FINAL: Ensure image-only groups are not left behind when there are other
    # groups to attach into. This covers cases where no text-only groups were
    # present earlier but some image-only groups remain; the user requested
    # these be merged into relative groups when possible.
    img_only_idxs = [
        i
        for i, g in enumerate(groups_work)
        if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0
    ]
    if img_only_idxs:
        all_idxs = [i for i, _ in enumerate(groups_work)]
        # prefer targets that have text content
        text_targets = [
            i for i, g in enumerate(groups_work) if len(g.get("texts", [])) > 0
        ]
        # if no text targets, allow any other group
        for ii in list(sorted(img_only_idxs, reverse=True)):
            if len(groups_work[ii].get("images", [])) == 0:
                continue
            src = groups_work[ii]
            candidates = [i for i in all_idxs if i != ii]
            # prefer text-containing groups first
            if text_targets:
                candidates = [i for i in candidates if i in text_targets] or candidates

            best_j = None
            best_d = None
            for cj in candidates:
                tgt = groups_work[cj]
                d = rect_distance(src["bbox"], tgt["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = cj

            if best_j is not None:
                tgt = groups_work[best_j]
                tgt["_forced_image_attachments"] = (
                    tgt.get("_forced_image_attachments", 0) + 1
                )
                tgt["images"].extend(src.get("images", []))
                tgt["bbox"] = [
                    min(tgt["bbox"][0], src["bbox"][0]),
                    min(tgt["bbox"][1], src["bbox"][1]),
                    max(tgt["bbox"][2], src["bbox"][2]),
                    max(tgt["bbox"][3], src["bbox"][3]),
                ]
                src["images"] = []
                attachments += 1

        groups_work = [
            g
            for g in groups_work
            if not (len(g.get("images", [])) == 0 and len(g.get("texts", [])) == 0)
        ]

    try:
        OVERLAP_FRAC = float(os.getenv("PDF_GROUP_MERGE_OVERLAP_FRAC", "0.2"))
    except Exception:
        OVERLAP_FRAC = 0.2
    groups_work = _merge_whole_groups(groups_work, overlap_frac=OVERLAP_FRAC)
    return groups_work, attachments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input extractor JSON")
    ap.add_argument("--out", "-o", help="Output JSON", default=None)
    ap.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=5.0,
        help="Proximity threshold in pixels (euclidean)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print("Input not found:", args.input)
        raise SystemExit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = args.out
    if out_path is None:
        inp = args.input
        base_name = os.path.splitext(os.path.basename(inp))[0]
        out_dir = os.path.dirname(inp) or "."
        out_path = os.path.join(out_dir, f"{base_name}.grouped.fullblocks.json")

    pages = data.get("pages", [])
    total_extra_attachments = 0
    pages_changed = 0
    layout_stats = {}

    for page in pages:
        groups = group_page(page, threshold=args.threshold)
        layout = page.get("_detected_layout", "unknown")
        layout_stats[layout] = layout_stats.get(layout, 0) + 1

        # attach remaining lone texts safely
        new_groups, att = attach_remaining_groups(
            groups, threshold=args.threshold, page=page
        )
        page["groups"] = new_groups
        total_extra_attachments += att
        if att > 0:
            pages_changed += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Wrote grouped JSON to:", out_path)
    print(
        f"Layout detection: {', '.join(f'{k}={v}' for k, v in sorted(layout_stats.items()))}"
    )
    print(
        f"Extra safe attachments made by post-processor: {total_extra_attachments} on {pages_changed} pages"
    )


if __name__ == "__main__":
    main()
