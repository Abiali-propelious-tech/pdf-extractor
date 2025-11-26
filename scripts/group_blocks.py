#!/usr/bin/env python3
"""
Group image and text blocks by proximity from extractor JSON.

This is a copy moved into scripts/ for organizational purposes.
"""
import json
import argparse
import os
import math
from typing import List, Tuple


def rect_distance(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 < bx0:
        dx = bx0 - ax1
    elif bx1 < ax0:
        dx = ax0 - bx1
    else:
        dx = 0
    if ay1 < by0:
        dy = by0 - ay1
    elif by1 < ay0:
        dy = ay0 - by1
    else:
        dy = 0
    return math.hypot(dx, dy)


def rects_overlap(a: List[float], b: List[float]) -> bool:
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
    images = page.get("images", [])
    texts = page.get("texts", [])

    if not images or not texts:
        return "scattered"

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
        if len(texts) == 0:
            return "full_page_single"
        return "vertical_flow"

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

    page_area = page_width * page_height
    density = total_elements / (page_area / 10000) if page_area > 0 else 0

    if total_elements <= 3 and avg_img_width > page_width * 0.5:
        return "full_page_single"

    if (
        num_images >= 4
        and img_span_x > page_width * 0.6
        and avg_img_width > page_width * 0.25
    ):
        img_widths_similar = (max(img_widths) - min(img_widths)) < avg_img_width * 0.3
        if img_widths_similar:
            return "catalog_grid"

    if num_images >= 6:
        img_widths_similar = (max(img_widths) - min(img_widths)) < avg_img_width * 0.4
        y_positions = sorted(set(round(cy / 50) * 50 for cy in img_centers_y))
        x_positions = sorted(set(round(cx / 50) * 50 for cx in img_centers_x))
        if len(y_positions) >= 2 and len(x_positions) >= 2 and img_widths_similar:
            return "table_grid"

    left_third = page_left + page_width * 0.35
    right_third = page_left + page_width * 0.65

    images_on_left = sum(1 for cx in img_centers_x if cx < left_third)
    images_on_right = sum(1 for cx in img_centers_x if cx > right_third)

    if (
        images_on_left > len(img_centers_x) * 0.75
        or images_on_right > len(img_centers_x) * 0.75
    ):
        return "sidebar"

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

    if num_texts > num_images * 3:
        text_x_centers = [
            (txt.get("bbox", [0, 0, 0, 0])[0] + txt.get("bbox", [0, 0, 0, 0])[2]) / 2.0
            for txt in texts
            if len(txt.get("bbox", [])) == 4
        ]
        if text_x_centers:
            x_buckets = {}
            bucket_size = page_width / 10
            for x in text_x_centers:
                bucket = int(x / bucket_size)
                x_buckets[bucket] = x_buckets.get(bucket, 0) + 1
            significant_buckets = [
                b for b, count in x_buckets.items() if count > num_texts * 0.15
            ]
            if len(significant_buckets) >= 2:
                return "multi_column"

    if img_span_y > page_height * 0.5 and img_span_x < page_width * 0.4:
        return "vertical_flow"

    if img_span_x > page_width * 0.7 and img_span_y < page_height * 0.3:
        return "horizontal_strips"

    if density > 5.0:
        return "scattered_dense"
    elif density < 1.5:
        return "scattered_sparse"

    if num_images >= 4:
        distances = []
        for i in range(len(img_centers_x)):
            for j in range(i + 1, len(img_centers_x)):
                dx = img_centers_x[i] - img_centers_x[j]
                dy = img_centers_y[i] - img_centers_y[j]
                distances.append(math.sqrt(dx * dx + dy * dy))
        if distances:
            avg_dist = sum(distances) / len(distances)
            variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
            if variance > avg_dist * avg_dist:
                return "mixed_density"

    return "scattered_sparse"


def _element_items(page: dict):
    """Yield (type, id, bbox, blocked, raw_obj) for each page element"""
    for img in page.get("images", []):
        yield (
            "image",
            img.get("id"),
            img.get("bbox", []),
            img.get("blocked", False),
            img,
        )
    for txt in page.get("texts", []):
        yield (
            "text",
            txt.get("id"),
            txt.get("bbox", []),
            txt.get("blocked", False),
            txt,
        )


def _bbox_union(bboxes: List[List[float]]) -> List[float]:
    x0s = [b[0] for b in bboxes]
    y0s = [b[1] for b in bboxes]
    x1s = [b[2] for b in bboxes]
    y1s = [b[3] for b in bboxes]
    return [min(x0s), min(y0s), max(x1s), max(y1s)]


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

    # Collect nodes and maps for full objects: prefix with kind to avoid id clashes
    images = []
    texts = []
    images_map = {}
    texts_map = {}
    for im in page.get("images", []):
        bbox = im.get("bbox", [])
        if len(bbox) == 4:
            key = f"img:{im.get('id')}"
            images_map[key] = im
            images.append((key, bbox))
    for tb in page.get("texts", []):
        bbox = tb.get("bbox", [])
        if len(bbox) == 4:
            # Skip text elements explicitly marked as blocked during primary grouping
            # so 'blocked' captions/labels won't create spurious groups.
            # They will be considered later by attach_remaining_groups which
            # can attach them to nearby groups when appropriate.
            if tb.get("blocked", False):
                continue
            key = f"txt:{tb.get('id')}"
            texts_map[key] = tb
            texts.append((key, bbox))

    dsu = DSU()

    # Step 1: group text-to-text using the chosen mode (preserve multi-line caption grouping)
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
                        ax0 - threshold,
                        ay0 - threshold,
                        ax1 + threshold,
                        ay1 + threshold,
                    )
                    b_ex = (
                        bx0 - threshold,
                        by0 - threshold,
                        bx1 + threshold,
                        by1 + threshold,
                    )
                    overlap = not (
                        a_ex[2] < b_ex[0]
                        or b_ex[2] < a_ex[0]
                        or a_ex[3] < b_ex[1]
                        or b_ex[3] < a_ex[1]
                    )
                    if overlap:
                        dsu.union(a_key, b_key)
                else:
                    dist = rect_distance(a_bbox, b_bbox)
                    if dist <= threshold:
                        dsu.union(a_key, b_key)
            except Exception:
                dist = rect_distance(a_bbox, b_bbox)
                if dist <= threshold:
                    dsu.union(a_key, b_key)

    # Step 2: merge images that overlap (so nearby small slices become one anchor)
    in_ = len(images)
    # compute average image size to avoid merging images that are actually
    # distinct (e.g., grid items separated by gutters). We'll only merge
    # if the expanded boxes overlap OR the center distance is comparatively tiny
    # relative to the average image size.
    img_ws = [b[2] - b[0] for _, b in images] if images else [0.0]
    img_hs = [b[3] - b[1] for _, b in images] if images else [0.0]
    avg_img_w = sum(img_ws) / len(img_ws) if img_ws else 0.0
    avg_img_h = sum(img_hs) / len(img_hs) if img_hs else 0.0
    merge_close_frac = 0.45
    # For image-image merging we use a smaller padding than the general
    # threshold used for text-image grouping. Grid gutters are often small
    # and using the full `threshold` here caused adjacent tiles to be
    # considered overlapping. Use a reduced `img_pad` so that images are
    # only merged when truly contiguous.
    img_pad = max(1.0, threshold * 0.4)
    for i in range(in_):
        for j in range(i + 1, in_):
            a_key, a_bbox = images[i]
            b_key, b_bbox = images[j]
            ax0, ay0, ax1, ay1 = a_bbox
            bx0, by0, bx1, by1 = b_bbox
            a_ex = (ax0 - img_pad, ay0 - img_pad, ax1 + img_pad, ay1 + img_pad)
            b_ex = (bx0 - img_pad, by0 - img_pad, bx1 + img_pad, by1 + img_pad)
            overlap = not (
                a_ex[2] < b_ex[0]
                or b_ex[2] < a_ex[0]
                or a_ex[3] < b_ex[1]
                or b_ex[3] < a_ex[1]
            )
            # accept union if expanded boxes overlap (original behaviour)
            # or if the centroids are *very* close compared to the average
            # image size (protects grid gutters and preserves separate
            # product tiles).
            if overlap:
                dsu.union(a_key, b_key)
            else:
                # fallback: very small center distance relative to avg size
                acx = (a_bbox[0] + a_bbox[2]) / 2.0
                acy = (a_bbox[1] + a_bbox[3]) / 2.0
                bcx = (b_bbox[0] + b_bbox[2]) / 2.0
                bcy = (b_bbox[1] + b_bbox[3]) / 2.0
                cx_dist = abs(acx - bcx)
                cy_dist = abs(acy - bcy)
                if (
                    cx_dist <= avg_img_w * merge_close_frac
                    and cy_dist <= avg_img_h * merge_close_frac
                ):
                    dsu.union(a_key, b_key)

    # Step 3: Anchor texts to nearest/overlapping image
    # For each text (or text-component) prefer images whose expanded bbox overlaps;
    # otherwise pick nearest image within radius. If still none, try directional
    # heuristics (text to the right/below/left/top of image with small gap and
    # reasonable alignment) to attach captions placed beside or under images.
    radius_multiplier = 2.0
    directional_multiplier = 3.0
    align_frac = 0.6
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
        if best_img is None:
            for i_key, i_bbox in images:
                ix0, iy0, ix1, iy1 = i_bbox
                # text is to the right
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
                # text is to the left
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
                # text is below
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
                # text is above
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

        if best_img:
            dsu.union(t_key, best_img)

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

    # Before post-processing, split overly-large image-heavy groups so
    # separate thumbnails don't get merged into a single super-group.
    refined_groups = []
    gid_counter = 0
    for g in groups:
        if len(g.get("images", [])) > 1 and len(g.get("texts", [])) <= max(
            1, len(g.get("images", [])) // 2
        ):
            # split into smaller clusters where appropriate
            subs = _split_image_heavy_group(g)
            for sub in subs:
                gid_counter += 1
                # preserve any existing texts mapping from sub
                sub_id = f"group_{gid_counter}"
                sub["id"] = sub_id
                refined_groups.append(sub)
        else:
            gid_counter += 1
            g["id"] = f"group_{gid_counter}"
            refined_groups.append(g)

    groups = refined_groups

    # Post-processing pass: try to attach remaining text-only groups to nearby
    # image-only groups using a larger search radius. This helps when captions
    # are slightly further away than the base threshold.
    # Build index lists
    text_only_idxs = []
    # previously we only considered pure image-only groups as candidates for
    # attaching text-only groups. That left many text-only groups when images
    # had already been grouped with other texts. Treat ANY group containing
    # images as a candidate target so nearby text-only groups will attach to
    # image-bearing groups (improves association in dense layouts / sidebars).
    img_candidate_idxs = []
    for i, g in enumerate(groups):
        if len(g.get("images", [])) == 0 and len(g.get("texts", [])) > 0:
            text_only_idxs.append(i)
        # consider any group that already contains images as a valid target
        # (this includes groups that may already contain text). This ensures
        # that stray text-only groups can join their nearest image group.
        if len(g.get("images", [])) > 0:
            img_candidate_idxs.append(i)

    merged = set()
    large_multiplier = 4.0
    for ti in text_only_idxs:
        if ti in merged:
            continue
        tgroup = groups[ti]
        best_j = None
        best_d = None
        for ji in img_candidate_idxs:
            if ji in merged:
                continue
            # don't attempt to merge a text-only group into itself
            if ji == ti:
                continue
            ig = groups[ji]
            d = rect_distance(tgroup["bbox"], ig["bbox"])
            if best_d is None or d < best_d:
                best_d = d
                best_j = ji
        if (
            best_j is not None
            and best_d is not None
            and best_d <= threshold * large_multiplier
        ):
            # Decide merge direction: attach the text-only group INTO the
            # found image-bearing group (preferred) so product anchors stay
            # image-centric. This prevents creating many isolated text-only
            # groups when images already represent a product anchor.
            ig = groups[best_j]
            # append texts from text-only group into the selected image group
            ig.setdefault("texts", []).extend(tgroup.get("texts", []))
            # recompute bbox
            all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
            all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
            all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
            all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
            ig["bbox"] = [all_x0, all_y0, all_x1, all_y1]
            # mark the text-only index as merged so it will be removed
            merged.add(ti)

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
    # Start with a lower multiplier for iterative matching — we prefer
    # local attachments first and only expand if necessary. Using a
    # lower start reduces accidental far-away attachment and avoids
    # combining many images into a single group.
    start_mult = 2.0
    max_multiplier = 4096.0
    # outer loop: keep trying until no more lone pairs can be matched
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

        matched_any = False
        mult = start_mult
        # try increasing radii until we find at least one match or hit max
        while mult <= max_multiplier and not matched_any:
            attach_pairs = []  # list of (img_idx, txt_idx)
            used_txt = set()
            for ii in lone_img_idxs:
                if ii >= len(groups_work):
                    continue
                img_g = groups_work[ii]
                ix0, iy0, ix1, iy1 = img_g["bbox"]
                icx = (ix0 + ix1) / 2.0
                icy = (iy0 + iy1) / 2.0
                iw = max(1.0, ix1 - ix0)
                ih = max(1.0, iy1 - iy0)
                best_j = None
                best_d = None
                for j in lone_txt_idxs:
                    if j in used_txt or j >= len(groups_work):
                        continue
                    txt_g = groups_work[j]
                    tx0, ty0, tx1, ty1 = txt_g["bbox"]
                    tcx = (tx0 + tx1) / 2.0
                    tcy = (ty0 + ty1) / 2.0
                    dx = tcx - icx
                    dy = tcy - icy
                    # determine cardinal direction: prefer the axis with larger magnitude
                    if abs(dx) >= abs(dy):
                        # candidate is left/right
                        d = rect_distance(img_g["bbox"], txt_g["bbox"])
                        # vertical alignment check
                        vert_overlap = max(0, min(iy1, ty1) - max(iy0, ty0))
                        if (
                            vert_overlap >= ih * (align_frac / 2)
                            or abs(tcy - icy) <= ih * align_frac
                        ):
                            if d <= threshold * mult and (best_d is None or d < best_d):
                                best_d = d
                                best_j = j
                    else:
                        # candidate is top/bottom
                        d = rect_distance(img_g["bbox"], txt_g["bbox"])
                        horiz_overlap = max(0, min(ix1, tx1) - max(ix0, tx0))
                        if (
                            horiz_overlap >= iw * (align_frac / 2)
                            or abs(tcx - icx) <= iw * align_frac
                        ):
                            if d <= threshold * mult and (best_d is None or d < best_d):
                                best_d = d
                                best_j = j

                if best_j is not None:
                    attach_pairs.append((ii, best_j))
                    used_txt.add(best_j)

            if attach_pairs:
                # perform attachments (merge texts into image groups), remove text groups
                remove_idxs = set()
                for ii, jj in attach_pairs:
                    if ii >= len(groups_work) or jj >= len(groups_work):
                        continue
                    img_g = groups_work[ii]
                    txt_g = groups_work[jj]
                    img_g["texts"].extend(txt_g.get("texts", []))
                    # recompute bbox
                    all_x0 = min(img_g["bbox"][0], txt_g["bbox"][0])
                    all_y0 = min(img_g["bbox"][1], txt_g["bbox"][1])
                    all_x1 = max(img_g["bbox"][2], txt_g["bbox"][2])
                    all_y1 = max(img_g["bbox"][3], txt_g["bbox"][3])
                    img_g["bbox"] = [all_x0, all_y0, all_x1, all_y1]
                    remove_idxs.add(jj)
                # remove absorbed text groups in descending order of index
                for idx in sorted(remove_idxs, reverse=True):
                    if 0 <= idx < len(groups_work):
                        groups_work.pop(idx)
                matched_any = True
            else:
                mult *= 2.0

        if not matched_any:
            break

    return groups_work


def _split_image_heavy_group(g: dict) -> List[dict]:
    """If a group contains many images but few/no texts, split the
    image cluster into multiple groups based on horizontal gaps. This
    prevents a whole page of thumbnails becoming a single super-group.
    Returns a list of groups (possibly 1 — the original group)"""
    imgs = g.get("images", [])
    texts = g.get("texts", [])
    if len(imgs) <= 1:
        return [g]

    # If there are plenty of texts (>= images/2) don't split — assume
    # the current mapping is okay and texts anchor the images.
    if len(texts) >= max(1, len(imgs) // 2):
        return [g]

    # compute centers and sort by x
    centers = []
    for im in imgs:
        x0, y0, x1, y1 = im.get("bbox", [0, 0, 0, 0])
        centers.append(((x0 + x1) / 2.0, im))
    centers.sort(key=lambda x: x[0])

    xs = [c[0] for c in centers]
    # estimate avg image width to detect 'big' gaps
    widths = [
        (im.get("bbox")[2] - im.get("bbox")[0]) for _, im in centers if im.get("bbox")
    ]
    avg_w = sum(widths) / len(widths) if widths else 0.0
    # Use a slightly tighter splitting threshold so we favor splitting
    # clusters of neighboring thumbnails into separate groups when there
    # are moderate gutters between items (this avoids super-groups across
    # several catalog tiles).
    gap_threshold = max(1.0, avg_w * 0.45)

    clusters = []
    cur_cluster = [centers[0][1]]
    for i in range(1, len(centers)):
        gap = centers[i][0] - centers[i - 1][0]
        if gap > gap_threshold:
            clusters.append(cur_cluster)
            cur_cluster = [centers[i][1]]
        else:
            cur_cluster.append(centers[i][1])
    clusters.append(cur_cluster)

    # Build new groups from clusters — preserve texts in original group
    out = []
    for cluster in clusters:
        xs = [b[0] for b in [im.get("bbox") for im in cluster]]
        ys = [b[1] for b in [im.get("bbox") for im in cluster]]
        x1s = [b[2] for b in [im.get("bbox") for im in cluster]]
        y1s = [b[3] for b in [im.get("bbox") for im in cluster]]
        bbox = [min(xs), min(ys), max(x1s), max(y1s)]
        out.append({"id": None, "bbox": bbox, "images": cluster, "texts": []})

    # if there were texts in original group, try to attach each text to the nearest
    # new cluster based on center distance
    if texts:
        # compute centers for clusters
        cluster_centers = []
        for c in out:
            x0, y0, x1, y1 = c.get("bbox")
            cluster_centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        for t in texts:
            tx0, ty0, tx1, ty1 = t.get("bbox", [0, 0, 0, 0])
            tcx = (tx0 + tx1) / 2.0
            tcy = (ty0 + ty1) / 2.0
            best_i = None
            best_d = None
            for i, cc in enumerate(cluster_centers):
                dx = tcx - cc[0]
                dy = tcy - cc[1]
                d = math.hypot(dx, dy)
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                out[best_i].setdefault("texts", []).append(t)

    return out


def attach_remaining_groups(
    groups: List[dict], threshold: float, page: dict
) -> Tuple[List[dict], int]:
    """Attach remaining elements (usually blocked items) into the nearest existing groups.

    Returns (new_groups, extra_count) where extra_count is the number of items attached.
    """
    # Build an index of IDs already present
    grouped_ids = set()
    for g in groups:
        for img in g.get("images", []):
            if isinstance(img, dict):
                grouped_ids.add(img.get("id"))
        for t in g.get("texts", []):
            if isinstance(t, dict):
                grouped_ids.add(t.get("id"))

    # Build simple bbox for each group
    for g in groups:
        if "bbox" not in g and g.get("bboxes"):
            g["bbox"] = _bbox_union(g.get("bboxes", []))
        elif "bbox" not in g:
            g["bbox"] = [0, 0, 0, 0]

    extras = 0

    def attach_one(item_raw, kind):
        nonlocal extras
        bbox = item_raw.get("bbox", [])
        # Find best group by distance
        best_g = None
        best_d = None
        for g in groups:
            d = rect_distance(g.get("bbox", [0, 0, 0, 0]), bbox)
            if best_d is None or d < best_d:
                best_d = d
                best_g = g
        if best_g is not None and best_d is not None and best_d <= threshold:
            # For blocked text items be stricter before attaching: only_number,
            # font_too_small or missing fonts tend to be decorative/labels and
            # should only be attached if they actually overlap the group
            # bbox (indicates they belong to it). Non-blocked items or
            # blocked items with other reasons can fall back to distance.
            if kind == "text" and item_raw.get("blocked", False):
                reason = item_raw.get("blocked_reason", "")
                # treat highly noisy or numeric-only texts carefully
                if reason in ("only_number", "font_too_small", "font_missing"):
                    if not rects_overlap(best_g.get("bbox", [0, 0, 0, 0]), bbox):
                        # refuse to attach noisy blocked text by distance alone
                        return False
            if kind == "image":
                best_g.setdefault("images", []).append(item_raw)
            else:
                best_g.setdefault("texts", []).append(item_raw)
            # expand group bbox
            best_g["bbox"] = _bbox_union([best_g.get("bbox", [0, 0, 0, 0]), bbox])
            extras += 1
            return True
        # fallback for text: try column/header alignment attachment — if a
        # text block horizontally overlaps several image-groups (e.g. a
        # column header or color swatch label), attach it to those image
        # groups instead of leaving the images unlabelled.
        if kind == "text":
            tb = item_raw.get("bbox", [])
            if len(tb) == 4:
                tx0, ty0, tx1, ty1 = tb
                tw = max(1.0, tx1 - tx0)
                # do not attempt wide page headers
                page_w = page.get("width") or None
                if page_w and tw >= page_w * 0.9:
                    return False

                candidates = []
                for g in groups:
                    # only consider groups that contain images as targets
                    if not g.get("images"):
                        continue
                    gx0, gy0, gx1, gy1 = g.get("bbox", [0, 0, 0, 0])
                    # fraction of text width overlapping group's x-range
                    overlap_x = max(0.0, min(tx1, gx1) - max(tx0, gx0))
                    frac = overlap_x / tw
                    # prefer text just above group or slightly overlapping
                    is_above = ty1 <= gy0 + max(4.0, threshold * 2)
                    if frac >= 0.25 and is_above:
                        candidates.append(g)

                if candidates:
                    for c in candidates:
                        c.setdefault("texts", []).append(item_raw)
                        c["bbox"] = _bbox_union([c.get("bbox", [0, 0, 0, 0]), tb])
                        extras += 1
                    return True

        return False

    # Find remaining elements from the page and attempt to attach
    for img in page.get("images", []):
        if img.get("id") in grouped_ids:
            continue
        attached = attach_one(img, "image")
        if not attached:
            groups.append({"images": [img], "texts": [], "bbox": img.get("bbox", [])})

    for txt in page.get("texts", []):
        if txt.get("id") in grouped_ids:
            continue
        # attach small or blocked text to nearest group if possible
        attached = attach_one(txt, "text")
        if not attached:
            # If the text is explicitly 'blocked' (small font, numeric-only,
            # etc.) we prefer to NOT create a standalone text-only group for it
            # because these frequently flood the grouping output and reduce
            # downstream grouping precision. Skip creating a new group for
            # blocked items; unblocked text should still create a group so
            # it remains discoverable.
            if txt.get("blocked"):
                # do not create a new group for blocked text — leave it ungrouped
                continue
            groups.append({"images": [], "texts": [txt], "bbox": txt.get("bbox", [])})

    # ------------------------------------------------------------------
    # Legacy-style iterative attachments (safe fallback)
    # The original top-level `attach_remaining.process_page` used a
    # conservative scoring-based pass that iteratively attached small
    # text-only groups to nearby image-bearing groups. During the cleanup
    # this behaviour was tightened and some previously-attached captions
    # remained unassociated. Re-run a bounded iterative pass here to
    # recover those legitimate attachments while preserving overlap and
    # text-count safety checks.
    # ------------------------------------------------------------------
    MAX_TEXTS_PER_GROUP = 7
    MAX_CHARS_PER_GROUP = 800
    TEXT_COUNT_WEIGHT = 10.0
    CHAR_COUNT_WEIGHT = 0.01
    FINAL_MULT = 8.0

    # Iteratively try to attach lone text-only groups to image-bearing
    # candidate targets. This mirrors the behaviour of the older helper
    # and can re-attach ad-hoc captions that survived earlier passes.
    while True:
        made_any = False
        text_idxs = [
            i
            for i, g in enumerate(groups)
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        ]
        cand_idxs = [i for i, g in enumerate(groups) if len(g.get("images", [])) > 0]
        if not text_idxs or not cand_idxs:
            break

        # sort by small groups first to attach least informative fragments
        def group_chars(g):
            tot = 0
            for t in g.get("texts", []):
                if isinstance(t, dict):
                    s = t.get("content") or t.get("text") or ""
                else:
                    s = str(t)
                if isinstance(s, str):
                    tot += len(s)
            return tot

        text_order = sorted(
            text_idxs,
            key=lambda i: (len(groups[i].get("texts", [])), group_chars(groups[i])),
        )

        removed = set()
        for ti in text_order:
            if ti in removed:
                continue
            src = groups[ti]
            best = None
            best_score = None
            for cj in cand_idxs:
                if cj == ti:
                    continue
                tgt = groups[cj]
                # capacity checks (texts/chars)
                if (
                    len(tgt.get("texts", [])) + len(src.get("texts", []))
                    > MAX_TEXTS_PER_GROUP
                ):
                    continue
                target_chars = group_chars(tgt)
                src_chars = group_chars(src)
                if target_chars + src_chars > MAX_CHARS_PER_GROUP:
                    continue

                all_x0 = min(tgt["bbox"][0], src["bbox"][0])
                all_y0 = min(tgt["bbox"][1], src["bbox"][1])
                all_x1 = max(tgt["bbox"][2], src["bbox"][2])
                all_y1 = max(tgt["bbox"][3], src["bbox"][3])
                new_bbox = [all_x0, all_y0, all_x1, all_y1]

                # avoid attachments that would cause overlap with unrelated groups
                overlaps = False
                for k, other in enumerate(groups):
                    if k == ti or k == cj:
                        continue
                    if rects_overlap(new_bbox, other.get("bbox", [0, 0, 0, 0])):
                        overlaps = True
                        break
                if overlaps:
                    continue

                d = rect_distance(
                    tgt.get("bbox", [0, 0, 0, 0]), src.get("bbox", [0, 0, 0, 0])
                )
                if d > threshold * FINAL_MULT:
                    continue

                score = (
                    d
                    + TEXT_COUNT_WEIGHT * len(tgt.get("texts", []))
                    + CHAR_COUNT_WEIGHT * target_chars
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = cj

            if best is not None:
                # attach and mark for removal
                tgt = groups[best]
                tgt.setdefault("texts", []).extend(src.get("texts", []))
                tgt["bbox"] = _bbox_union(
                    [tgt.get("bbox", [0, 0, 0, 0]), src.get("bbox", [0, 0, 0, 0])]
                )
                removed.add(ti)
                extras += 1
                made_any = True

        if not made_any:
            break

        # remove absorbed groups
        for idx in sorted(removed, reverse=True):
            if 0 <= idx < len(groups):
                groups.pop(idx)

    return groups, extras
