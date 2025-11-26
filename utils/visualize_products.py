"""
Product Bounding Box Visualizer

This script creates a visualization of all product images on the full PDF page
by drawing colored bounding boxes around each product. Each box color is mapped
to the corresponding image file in the XML structure.
"""

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import os
import json
import colorsys


def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        # Use high saturation and value for vibrant colors
        saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.9 - (i % 2) * 0.1  # Vary brightness slightly
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_bounding_boxes(xml_path, output_folder, line_width=3, update_xml=True):
    """
    Draw colored bounding boxes on the full page image for each product.

    Args:
        xml_path: Path to the XML structure file
        output_folder: Path to the output folder containing images
        line_width: Width of the bounding box lines (default: 3)
        update_xml: Whether to update the XML file with color and index info (default: True)

    Returns:
        List of all color mappings for all pages
    """
    print(f"📖 Reading XML from: {xml_path}")

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_mappings = []  # Collect mappings from all pages

    # Process each page
    for page in root.findall("page"):
        page_num = page.get("number")
        print(f"\n📄 Processing page {page_num}")

        # Get page dimensions
        page_width = float(page.get("width", 0))
        page_height = float(page.get("height", 0))
        print(f"   Page dimensions: {page_width} x {page_height}")

        # Find all image blocks (excluding extraction_failed ones)
        image_blocks = []
        for img_block in page.findall("image_block"):
            img_file = img_block.get("file")
            # Skip extraction_failed and full page images
            if img_file and img_file != "extraction_failed" and "full" not in img_file:
                bbox = img_block.get("bbox")
                if bbox:
                    coords = [float(x) for x in bbox.split(",")]
                    image_blocks.append(
                        {
                            "id": img_block.get("id"),
                            "file": img_file,
                            "bbox": coords,  # [x0, y0, x1, y1]
                            "width": img_block.get("width"),
                            "height": img_block.get("height"),
                        }
                    )

        print(f"   Found {len(image_blocks)} product images")

        if not image_blocks:
            print("   ⚠️  No product images found on this page")
            continue

        # Load the full page image
        full_page_path = os.path.join(output_folder, f"page_{page_num}_full.png")
        if not os.path.exists(full_page_path):
            print(f"   ❌ Full page image not found: {full_page_path}")
            continue

        print(f"   📸 Loading full page image: {full_page_path}")
        full_img = Image.open(full_page_path)
        img_width, img_height = full_img.size

        # Calculate scaling factor (full page image is 2x resolution)
        scale_x = img_width / page_width
        scale_y = img_height / page_height
        print(f"   📏 Image scale: {scale_x:.2f}x, {scale_y:.2f}y")

        # Generate distinct colors for each product
        colors = generate_distinct_colors(len(image_blocks))

        # Create a color mapping
        color_mapping = []

        # Draw bounding boxes
        draw = ImageDraw.Draw(full_img)

        for idx, (img_block, color) in enumerate(zip(image_blocks, colors)):
            # Scale bbox coordinates to match image resolution
            x0, y0, x1, y1 = img_block["bbox"]
            scaled_bbox = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

            # Draw rectangle
            draw.rectangle(scaled_bbox, outline=color, width=line_width)

            # Add a small label with the index number
            try:
                # Try to use a decent font size
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
                # Try to load a font, fall back to default if not available
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        font_size,
                    )
                except:
                    font = ImageFont.load_default()

                # Draw the index number in the top-left corner of the box
                label = f"{idx + 1}"
                # Create a background for the text
                bbox_width = scaled_bbox[2] - scaled_bbox[0]
                bbox_height = scaled_bbox[3] - scaled_bbox[1]

                # Position label in top-left corner
                text_x = scaled_bbox[0] + 5
                text_y = scaled_bbox[1] + 5

                # Draw text with background
                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((text_x, text_y), label, fill="white", font=font)

            except Exception as e:
                print(f"      ⚠️  Could not add label: {e}")

            # Store color mapping
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
                f"      ✅ Box {idx + 1}: {os.path.basename(img_block['file'])} "
                f"at ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}) - Color: {color}"
            )

        # Update XML with color and index information
        if update_xml:
            print(f"\n   🔧 Updating XML with color and index information...")
            for map_item in color_mapping:
                # Find the corresponding image_block in XML by ID
                for img_block in page.findall("image_block"):
                    if img_block.get("id") == map_item["image_id"]:
                        # Add color and index attributes
                        img_block.set("visualization_index", str(map_item["index"]))
                        img_block.set("visualization_color", map_item["color"])
                        img_block.set(
                            "visualization_rgb",
                            f"{map_item['rgb'][0]},{map_item['rgb'][1]},{map_item['rgb'][2]}",
                        )
                        break

            # Save updated XML
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"   ✅ Updated XML with visualization data")

        # Save the annotated image
        output_path = os.path.join(output_folder, f"page_{page_num}_annotated.png")
        full_img.save(output_path)
        print(f"   ✅ Saved annotated image: {output_path}")

        # Add this page's mapping to the collection
        all_mappings.append(
            {
                "page": page_num,
                "mapping": color_mapping,
                "total_products": len(color_mapping),
            }
        )

    # Return all mappings from all pages
    return all_mappings



def main():
    """Main function to visualize products from all available PDFs"""
    import sys

    # Default paths
    output_base = "./output"

    # Check command line arguments
    if len(sys.argv) > 1:
        # User provided specific folder
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
        # Process all folders in output directory
        if not os.path.exists(output_base):
            print(f"❌ Output folder not found: {output_base}")
            return

        # Find all PDF output folders
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
    print("=" * 60)
    print("\n📂 Check the output folders for:")
    print("   • *_annotated.png - Full page with colored boxes")
    print("   • *_color_mapping.json - Color to image mapping data")
    print("   • *_legend.html - Interactive HTML legend with thumbnails")


if __name__ == "__main__":
    main()
