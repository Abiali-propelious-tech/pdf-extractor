import json, os, sys
from collections import defaultdict


def key_size(v, rd):
    if v is None:
        return "__NULL__"
    try:
        return round(float(v), rd)
    except Exception:
        return "__NULL__"


if __name__ == "__main__":
    base = "output/pdf_merged/pdf_merged_structure.fullblocks.grouped.json"
    if not os.path.exists(base):
        print("Grouped JSON not found at", base)
        sys.exit(1)
    rd = int(os.getenv("PDF_TEXT_SIZE_ROUND_DECIMALS", "0"))
    with open(base, "r", encoding="utf-8") as f:
        data = json.load(f)
    mixed = []
    for page in data.get("pages", []):
        pnum = page.get("page_number")
        buckets = defaultdict(set)
        for t in page.get("texts", []):
            ks = key_size(t.get("size"), rd)
            blocked = bool(t.get("blocked"))
            buckets[ks].add(blocked)
        for ks, flags in buckets.items():
            if len(flags) > 1:
                mixed.append((pnum, ks, sorted(list(flags))))
    if not mixed:
        print("No mixed blocked-state cases found")
    else:
        print(f"Found {len(mixed)} mixed blocked-state cases (page, key, flags):")
        for i, (p, ks, flags) in enumerate(mixed[:200]):
            print(p, ks, flags)
    # exit code >0 if mixed found
    sys.exit(0 if not mixed else 2)
