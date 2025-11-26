"""
Product Bounding Box Visualizer (moved to scripts/)
"""

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import os
import json
import colorsys


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + (i % 3) * 0.1
        value = 0.9 - (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_bounding_boxes(xml_path, output_folder, line_width=3, update_xml=True):
    print(f"📖 Reading XML from: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_mappings = []
    for page in root.findall("page"):
        page_num = page.get("number")
        print(f"\n📄 Processing page {page_num}")
        page_width = float(page.get("width", 0))
        page_height = float(page.get("height", 0))
        image_blocks = []
        for img_block in page.findall("image_block"):
            img_file = img_block.get("file")
            if img_file and img_file != "extraction_failed" and "full" not in img_file:
                bbox = img_block.get("bbox")
                if bbox:
                    coords = [float(x) for x in bbox.split(",")]
                    image_blocks.append(
                        {
                            "id": img_block.get("id"),
                            "file": img_file,
                            "bbox": coords,
                            "width": img_block.get("width"),
                            "height": img_block.get("height"),
                        }
                    )

        print(f"   Found {len(image_blocks)} product images")

        if not image_blocks:
            print("   ⚠️  No product images found on this page")
            continue

        full_page_path = os.path.join(output_folder, f"page_{page_num}_full.png")
        if not os.path.exists(full_page_path):
            print(f"   ❌ Full page image not found: {full_page_path}")
            continue

        print(f"   📸 Loading full page image: {full_page_path}")
        full_img = Image.open(full_page_path)
        img_width, img_height = full_img.size

        scale_x = img_width / page_width
        scale_y = img_height / page_height
        print(f"   📏 Image scale: {scale_x:.2f}x, {scale_y:.2f}y")

        colors = generate_distinct_colors(len(image_blocks))
        color_mapping = []
        draw = ImageDraw.Draw(full_img)

        for idx, (img_block, color) in enumerate(zip(image_blocks, colors)):
            x0, y0, x1, y1 = img_block["bbox"]
            scaled_bbox = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

            draw.rectangle(scaled_bbox, outline=color, width=line_width)

            try:
                font_size = max(
                    12,
                    int(
                        min(
                            scaled_bbox[2] - scaled_bbox[0],
                            scaled_bbox[3] - scaled_bbox[1],
                        )
                        * 0.1
                    ),
                )
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        font_size,
                    )
                except:
                    font = ImageFont.load_default()

                label = f"{idx + 1}"
                bbox_width = scaled_bbox[2] - scaled_bbox[0]
                bbox_height = scaled_bbox[3] - scaled_bbox[1]
                text_x = scaled_bbox[0] + 5
                text_y = scaled_bbox[1] + 5

                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((text_x, text_y), label, fill="white", font=font)

            except Exception as e:
                print(f"      ⚠️  Could not add label: {e}")

            color_mapping.append(
                {
                    "index": idx + 1,
                    "color": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "rgb": color,
                    "image_id": img_block["id"],
                    "image_file": os.path.basename(img_block["file"]),
                    "bbox": img_block["bbox"],
                    "width": img_block["width"],
                    "height": img_block["height"],
                }
            )

            print(
                f"      ✅ Box {idx + 1}: {os.path.basename(img_block['file'])} at ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}) - Color: {color}"
            )

        if update_xml:
            print(f"\n   🔧 Updating XML with color and index information...")
            for map_item in color_mapping:
                for img_block in page.findall("image_block"):
                    if img_block.get("id") == map_item["image_id"]:
                        img_block.set("visualization_index", str(map_item["index"]))
                        img_block.set("visualization_color", map_item["color"])
                        img_block.set(
                            "visualization_rgb",
                            f"{map_item['rgb'][0]},{map_item['rgb'][1]},{map_item['rgb'][2]}",
                        )
                        break
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"   ✅ Updated XML with visualization data")

        output_path = os.path.join(output_folder, f"page_{page_num}_annotated.png")
        full_img.save(output_path)
        print(f"   ✅ Saved annotated image: {output_path}")

        all_mappings.append(
            {
                "page": page_num,
                "mapping": color_mapping,
                "total_products": len(color_mapping),
            }
        )

    return all_mappings


def main():
    import sys

    output_base = "./output"

    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
        xml_file = os.path.join(
            pdf_folder, f"{os.path.basename(pdf_folder)}_structure.xml"
        )
        images_folder = os.path.join(pdf_folder, "images")

        if not os.path.exists(xml_file):
            print(f"❌ XML file not found: {xml_file}")
            return

        if not os.path.exists(images_folder):
            print(f"❌ Images folder not found: {images_folder}")
            return

        print(f"🎨 Visualizing products from: {pdf_folder}")
        draw_bounding_boxes(xml_file, images_folder)
    else:
        if not os.path.exists(output_base):
            print(f"❌ Output folder not found: {output_base}")
            return

        pdf_folders = [
            os.path.join(output_base, d)
            for d in os.listdir(output_base)
            if os.path.isdir(os.path.join(output_base, d))
        ]

        if not pdf_folders:
            print(f"❌ No PDF folders found in: {output_base}")
            return

        print(f"🎨 Found {len(pdf_folders)} PDF folders to process")

        for pdf_folder in pdf_folders:
            folder_name = os.path.basename(pdf_folder)
            xml_file = os.path.join(pdf_folder, f"{folder_name}_structure.xml")
            images_folder = os.path.join(pdf_folder, "images")

            if not os.path.exists(xml_file):
                print(f"⚠️  Skipping {folder_name}: XML not found")
                continue

            if not os.path.exists(images_folder):
                print(f"⚠️  Skipping {folder_name}: Images folder not found")
                continue

            print(f"\n{'='*60}")
            print(f"🎨 Processing: {folder_name}")
            print(f"{'='*60}")

            draw_bounding_boxes(xml_file, images_folder)

    print("\n" + "=" * 60)
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
