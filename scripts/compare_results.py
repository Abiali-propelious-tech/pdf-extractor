#!/usr/bin/env python3
"""Compare old vs new layout-aware grouping results (moved to scripts/)."""
import json
import sys

if len(sys.argv) < 3:
    print("Usage: compare_results.py <old_grouped.json> <new_grouped.json>")
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    old_data = json.load(f)
with open(sys.argv[2], "r", encoding="utf-8") as f:
    new_data = json.load(f)

old_pages = old_data.get("pages", [])
new_pages = new_data.get("pages", [])

print("=" * 80)
print("LAYOUT-AWARE GROUPING COMPARISON")
print("=" * 80)
print(f"\nTotal pages: {len(new_pages)}\n")

layout_counts = {}
for page in new_pages:
    layout = page.get("_detected_layout", "unknown")
    layout_counts[layout] = layout_counts.get(layout, 0) + 1

print("DETECTED LAYOUT DISTRIBUTION:")
print("-" * 80)
for layout, count in sorted(layout_counts.items(), key=lambda x: -x[1]):
    pct = (count / len(new_pages) * 100) if new_pages else 0
    print(f"  {layout:25s}: {count:3d} pages ({pct:5.1f}%)")

print("\n\nPER-PAGE IMPROVEMENTS:")
print("-" * 80)
print(
    f"{'Page':<6} {'Layout':<20} {'Old Groups':<12} {'New Groups':<12} {'Lone Texts Change':<20}"
)
print("-" * 80)

better = 0
worse = 0
same = 0

for i, (old_page, new_page) in enumerate(zip(old_pages, new_pages)):
    old_groups = old_page.get("groups", [])
    new_groups = new_page.get("groups", [])
    layout = new_page.get("_detected_layout", "unknown")

    old_text_only = sum(
        1
        for g in old_groups
        if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
    )
    new_text_only = sum(
        1
        for g in new_groups
        if len(g.get("texts", [])) > 0 and len(g.get("images", [])) == 0
    )

    change = new_text_only - old_text_only
    if change < 0:
        better += 1
        status = f"✓ {change:+d} (better)"
    elif change > 0:
        worse += 1
        status = f"✗ {change:+d} (worse)"
    else:
        same += 1
        status = "= (same)"

    if change != 0:
        print(
            f"{i+1:<6} {layout:<20} {len(old_groups):<12} {len(new_groups):<12} {status:<20}"
        )

print("-" * 80)
print(f"\nSummary: {better} better, {worse} worse, {same} same")

print("\n\nGROUP SIZE ANALYSIS:")
print("-" * 80)


def analyze_groups(groups, label):
    sizes = [len(g.get("texts", [])) for g in groups if len(g.get("texts", [])) > 0]
    if not sizes:
        print(f"{label}: No text groups")
        return

    avg = sum(sizes) / len(sizes)
    max_size = max(sizes)
    large_groups = sum(1 for s in sizes if s > 10)

    print(f"{label}:")
    print(f"  Total groups with text: {len(sizes)}")
    print(f"  Average texts per group: {avg:.1f}")
    print(f"  Max texts in a group: {max_size}")
    print(f"  Groups with >10 texts: {large_groups}")


all_old_groups = []
all_new_groups = []
for old_page, new_page in zip(old_pages, new_pages):
    all_old_groups.extend(old_page.get("groups", []))
    all_new_groups.extend(new_page.get("groups", []))

analyze_groups(all_old_groups, "OLD")
print()
analyze_groups(all_new_groups, "NEW")

print("\n" + "=" * 80)
