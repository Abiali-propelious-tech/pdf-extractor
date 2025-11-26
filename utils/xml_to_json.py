#!/usr/bin/env python3
"""
Convert extractor XML to a cleaner JSON structure.
Usage:
  python xml_to_json.py <xml_path> [--out json_path]

Output JSON schema:
{
  "source": "...",
  "total_pages": 1,
  "pages": [
    {
      "page_number": 1,
      "width": 960.0,
      "height": 540.0,
      "images": [ {id,bbox:[x0,y0,x1,y1], width, height, file} ... ],
      "texts": [ {id,bbox:[...], type, content, detected_type} ... ]
    }
  ]
}
"""
import xml.etree.ElementTree as ET
import json
import sys
import os
import re
from typing import List

PRICE_RE = re.compile(r"([₹€$£¥]|INR)\s*[0-9.,]+|[0-9][0-9,\.]+\s*(?:INR|Rs|rs|₹)")
SKU_RE = re.compile(r"#?\s*\d{5,8}")


def parse_bbox(bbox_str: str) -> List[float]:
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        return parts
    except Exception:
        return []


def detect_text_type(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if PRICE_RE.search(t):
        return "price"
    if SKU_RE.search(t):
        return "product_code"
    return ""


def xml_to_json(xml_path: str, out_path: str = None) -> str:
    if out_path is None:
        base = os.path.splitext(xml_path)[0]
        out_path = base + ".json"

    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {
        "source": root.get("source"),
        "total_pages": int(root.get("total_pages") or 0),
        "pages": [],
    }

    for page in root.findall("page"):
        page_num = int(page.get("number") or 0)
        pwidth = float(page.get("width") or 0)
        pheight = float(page.get("height") or 0)
        page_obj = {
            "page_number": page_num,
            "width": pwidth,
            "height": pheight,
            "images": [],
            "texts": [],
        }

        for img in page.findall("image_block"):
            bbox = parse_bbox(img.get("bbox") or "")
            page_obj["images"].append(
                {
                    "id": img.get("id"),
                    "bbox": bbox,
                    "width": float(img.get("width") or 0),
                    "height": float(img.get("height") or 0),
                    "file": img.get("file"),
                }
            )

        for tb in page.findall("text_block"):
            bbox = parse_bbox(tb.get("bbox") or "")
            text_content = tb.text or ""
            detected = detect_text_type(text_content)
            page_obj["texts"].append(
                {
                    "id": tb.get("id"),
                    "bbox": bbox,
                    "type": tb.get("type") or "",
                    "content": text_content.strip(),
                    "detected_type": detected,
                }
            )

        result["pages"].append(page_obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python xml_to_json.py <xml_path> [--out json_path]")
        sys.exit(1)
    xml_path = sys.argv[1]
    out = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--out":
        if len(sys.argv) >= 4:
            out = sys.argv[3]
    json_path = xml_to_json(xml_path, out)
    print("Wrote JSON to:", json_path)
