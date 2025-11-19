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
    for ti in text_only_idxs:
        if ti in merged:
            continue
        tgroup = groups[ti]
        best_j = None
        best_d = None
        for ji in img_only_idxs:
            if ji in merged:
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
            # merge ji into ti (attach image to the text group)
            ig = groups[best_j]
            # append images from ig into tgroup
            tgroup["images"].extend(ig.get("images", []))
            # recompute bbox
            xs = [
                tgroup["bbox"][0],
                tgroup["bbox"][1],
                tgroup["bbox"][2],
                tgroup["bbox"][3],
            ]
            xs = [
                tgroup["bbox"][0],
                tgroup["bbox"][1],
                tgroup["bbox"][2],
                tgroup["bbox"][3],
            ]
            all_x0 = min(tgroup["bbox"][0], ig["bbox"][0])
            all_y0 = min(tgroup["bbox"][1], ig["bbox"][1])
            all_x1 = max(tgroup["bbox"][2], ig["bbox"][2])
            all_y1 = max(tgroup["bbox"][3], ig["bbox"][3])
            tgroup["bbox"] = [all_x0, all_y0, all_x1, all_y1]
            merged.add(best_j)

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
        # If input path contains an `input/` folder segment, replace it with `output/`
        inp = args.input
        sep = os.sep
        replaced = False
        marker = sep + "input" + sep
        if marker in inp:
            out_path = inp.replace(marker, sep + "output" + sep)
            replaced = True
        elif inp.startswith("input" + sep):
            out_path = inp.replace("input" + sep, "output" + sep, 1)
            replaced = True
        else:
            inp_dir = os.path.dirname(inp) or "."
            base_name = os.path.splitext(os.path.basename(inp))[0]
            out_path = os.path.join(inp_dir, f"{base_name}.grouped.fullblocks.json")

        # normalize output filename to use a stable suffix
        if replaced:
            out_path = os.path.splitext(out_path)[0] + ".fullblocks.grouped.json"

    pages = data.get("pages", [])
    for page in pages:
        groups = group_page(page, threshold=args.threshold)
        page["groups"] = groups

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Wrote grouped JSON to:", out_path)


if __name__ == "__main__":
    main()
