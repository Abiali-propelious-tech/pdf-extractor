#!/usr/bin/env python3
"""Check detected layouts and grouping stats per page (moved to scripts/)."""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: check_layouts.py <grouped.json>")
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

pages = data.get("pages", [])
print(f"Total pages: {len(pages)}\n")

layout_counts = {}
for i, page in enumerate(pages):
    layout = page.get("_detected_layout", "unknown")
    groups = page.get("groups", [])

    layout_counts[layout] = layout_counts.get(layout, 0) + 1

    with_images = sum(1 for g in groups if len(g.get("images", [])) > 0)
    text_only = sum(
        1
        for g in groups
        if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
    )
    img_only = sum(
        1
        for g in groups
        if len(g.get("images", [])) > 0 and len(g.get("texts", [])) == 0
    )

    total_texts = sum(len(g.get("texts", [])) for g in groups)

    print(
        f"Page {i+1}: layout={layout}, groups={len(groups)}, with_img={with_images}, text_only={text_only}, img_only={img_only}, total_texts={total_texts}"
    )

    sorted_groups = sorted(groups, key=lambda g: len(g.get("texts", [])), reverse=True)[
        :3
    ]
    for j, g in enumerate(sorted_groups):
        n_img = len(g.get("images", []))
        n_txt = len(g.get("texts", []))
        if n_txt > 0 or n_img > 0:
            print(f"  Top group {j+1}: {n_img} imgs, {n_txt} texts")

print(f"\nLayout distribution:")
for layout, count in sorted(layout_counts.items()):
    print(f"  {layout}: {count} pages")
