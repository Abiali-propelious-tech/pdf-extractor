#!/usr/bin/env python3
"""Attach remaining lone text-only groups to image-containing groups when safe.

This file is a copy of the top-level attach_remaining.py moved to scripts/.
"""
import json
import sys
import math
from typing import List


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


def text_chars(g):
    tot = 0
    for t in g.get("texts", []):
        if isinstance(t, dict):
            s = t.get("content") or t.get("text") or ""
        else:
            s = str(t)
        if isinstance(s, str):
            tot += len(s)
    return tot


def text_count(g):
    return len(g.get("texts", []))


MAX_TEXTS_PER_GROUP = 7
MAX_CHARS_PER_GROUP = 800
TEXT_COUNT_WEIGHT = 10.0
CHAR_COUNT_WEIGHT = 0.01
THRESHOLD = 5.0
FINAL_MULT = 8.0


def process_page(groups):
    attachments = 0
    while True:
        made = False
        text_idxs = [
            i
            for i, g in enumerate(groups)
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        ]
        cand_idxs = [i for i, g in enumerate(groups) if len(g.get("images", [])) > 0]
        if not text_idxs or not cand_idxs:
            break
        text_order = sorted(
            text_idxs, key=lambda i: (text_count(groups[i]), text_chars(groups[i]))
        )
        removed = set()
        for ti in text_order:
            if ti in removed:
                continue
            src = groups[ti]
            best = None
            best_score = None
            best_d = None
            best_j = None
            for cj in cand_idxs:
                if cj == ti:
                    continue
                tgt = groups[cj]
                if text_count(tgt) + text_count(src) > MAX_TEXTS_PER_GROUP:
                    continue
                if text_chars(tgt) + text_chars(src) > MAX_CHARS_PER_GROUP:
                    continue
                all_x0 = min(tgt["bbox"][0], src["bbox"][0])
                all_y0 = min(tgt["bbox"][1], src["bbox"][1])
                all_x1 = max(tgt["bbox"][2], src["bbox"][2])
                all_y1 = max(tgt["bbox"][3], src["bbox"][3])
                new_bbox = [all_x0, all_y0, all_x1, all_y1]
                overlaps = False
                for k, other in enumerate(groups):
                    if k == ti or k == cj:
                        continue
                    if rects_overlap(new_bbox, other["bbox"]):
                        overlaps = True
                        break
                if overlaps:
                    continue
                d = rect_distance(tgt["bbox"], src["bbox"])
                if d > THRESHOLD * FINAL_MULT:
                    continue
                score = (
                    d
                    + TEXT_COUNT_WEIGHT * text_count(tgt)
                    + CHAR_COUNT_WEIGHT * text_chars(tgt)
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = cj
                    best_d = d
            if best is not None:
                tgt = groups[best]
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
            if 0 <= idx < len(groups):
                groups.pop(idx)
    return attachments


def main():
    if len(sys.argv) < 3:
        print("Usage: attach_remaining.py <input_grouped.json> <output_fixed.json>")
        raise SystemExit(1)
    inp = sys.argv[1]
    outp = sys.argv[2]
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_att = 0
    pages = data.get("pages", [])
    pages_changed = 0
    for i, page in enumerate(pages):
        groups = page.get("groups", [])
        before = sum(
            1
            for g in groups
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        )
        att = process_page(groups)
        after = sum(
            1
            for g in groups
            if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
        )
        if att > 0:
            pages_changed += 1
        total_att += att
        page["groups"] = groups
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote fixed grouped JSON to: {outp}")
    print(f"Total attachments made: {total_att}, pages changed: {pages_changed}")


if __name__ == "__main__":
    main()
