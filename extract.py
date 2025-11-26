# PDF to Structured Data Extractor with AI Enhancement
# This approach extracts exact coordinates and text positioning

import fitz  # pymupdf
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import re
from collections import defaultdict
import os
import base64
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import glob
import re
from utils.requester import graphql_requester
from utils.json_parser import parse_ai_json_response

load_dotenv()
# Optional OCR imports
try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
# AI/LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  AI features not available. Install: pip install langchain-google-genai")

# Optional grouping helper used to post-process JSON into grouped blocks
try:
    import utils.group_blocks as _group_blocks_module
except Exception:
    _group_blocks_module = None


class PDFStructureExtractor:
    """
    Extract PDF structure with precise coordinates
    Enhanced with AI for intelligent text-image matching
    """

    def __init__(self, gemini_api_key=None):
        print(f"==>> image_path: {os.path}")

        self.min_image_size = 50  # Skip tiny images/icons
        self.gemini_api_key = gemini_api_key
        # OCR availability
        self.ocr_available = OCR_AVAILABLE
        self.main_output_dir = None
        self.image_dir = None
        # Cache for sub-categories to avoid repeated API calls
        self.sub_categories_cache = {}
        # Initialize AI model if available and API key provided
        self.ai_model = None
        if AI_AVAILABLE and gemini_api_key:
            try:
                self.ai_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=gemini_api_key,
                    temperature=0.1,
                )
                print("✅ AI model initialized successfully with vision support")
            except Exception as e:
                print(f"⚠️  Failed to initialize AI model: {e}")
        elif not gemini_api_key:
            print("ℹ️  No Gemini API key provided. AI features disabled.")

    def create_output_structure(self, pdf_path, output_dir="output"):
        """Create organized output folder structure"""
        import os

        # Get PDF name without extension
        pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]

        # Create main output directory
        main_output_dir = os.path.join(output_dir, pdf_base)
        self.main_output_dir = main_output_dir
        # Create subdirectories
        images_dir = os.path.join(main_output_dir, "images")
        self.image_dir = images_dir
        # Create all directories
        os.makedirs(main_output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        return {
            "main_dir": main_output_dir,
            "images_dir": images_dir,
            "xml_path": os.path.join(main_output_dir, f"{pdf_base}_structure.xml"),
        }

    def pdf_to_xml(self, pdf_path, output_xml_path, image_folder=None):
        """Convert PDF to structured XML with coordinates"""
        doc = fitz.open(pdf_path)

        # Create root XML element
        root = ET.Element("catalogue")
        root.set("source", pdf_path)
        root.set("total_pages", str(len(doc)))

        # Use provided image folder or create default one
        if image_folder is None:
            pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
            image_folder = os.path.join(os.path.dirname(pdf_path), pdf_base)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Save a full-page snapshot as an image for review/analysis
            try:
                full_page_filename = os.path.join(
                    image_folder, f"page_{page_num+1}_full.png"
                )
                # Render at 2x for better readability
                mat = fitz.Matrix(2, 2)
                pix_full = page.get_pixmap(matrix=mat)
                pix_full.save(full_page_filename)
                if pix_full:
                    pix_full = None
            except Exception as e:
                print(f"⚠️  Could not save full-page image for page {page_num+1}: {e}")

            page_elem = self._extract_page_structure(page, page_num, image_folder)
            root.append(page_elem)
        # Save XML to provided path
        try:
            self._save_pretty_xml(root, output_xml_path)
            print(f"XML saved to: {output_xml_path}")
        except Exception as e:
            print(f"⚠️ Failed to save XML to {output_xml_path}: {e}")

        return output_xml_path

    def _extract_page_structure(self, page, page_num, image_folder):
        """Extract texts and images from a single page and return an XML element."""
        # Create page element
        page_elem = ET.Element("page")
        page_elem.set("number", str(page_num + 1))
        try:
            rect = page.rect
            page_elem.set("width", f"{rect.width:.1f}")
            page_elem.set("height", f"{rect.height:.1f}")
        except Exception:
            page_elem.set("width", "")
            page_elem.set("height", "")

        # Text extraction
        try:
            text_blocks = []
            # Use page.get_text("blocks") for blocks (x0,y0,x1,y1,text,block_no,...) if available
            try:
                blocks = page.get_text("blocks")
            except Exception:
                blocks = []

            # Try to extract richer text span data (font, size, color) using the dict output
            try:
                dict_output = page.get_text("dict")
            except Exception:
                dict_output = None

            for i, b in enumerate(blocks):
                try:
                    x0, y0, x1, y1, text, *_ = b
                except Exception:
                    continue
                tb = ET.Element("text_block")
                tb.set("id", f"text_{page_num}_{i}")
                tb.set("bbox", f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}")
                # Add width/height of text block
                try:
                    width = float(x1 - x0)
                    height = float(y1 - y0)
                    tb.set("width", f"{width:.1f}")
                    tb.set("height", f"{height:.1f}")
                except Exception:
                    tb.set("width", "")
                    tb.set("height", "")
                # Try enrich block information with font/size/color using spans if available
                font_name = ""
                font_size = ""
                font_color = ""
                try:
                    if dict_output and isinstance(dict_output, dict):
                        # Find a matching block by bbox (allow some tolerance)
                        tolerance = 1.0
                        dict_blocks = dict_output.get("blocks", [])
                        matched_block = None
                        for db in dict_blocks:
                            # only consider text blocks
                            if db.get("type") != 0:
                                continue
                            db_bbox = db.get("bbox") or []
                            if len(db_bbox) >= 4:
                                dx0, dy0, dx1, dy1 = db_bbox
                                if (
                                    abs(dx0 - x0) < tolerance
                                    and abs(dy0 - y0) < tolerance
                                    and abs(dx1 - x1) < tolerance
                                    and abs(dy1 - y1) < tolerance
                                ):
                                    matched_block = db
                                    break

                        # If not found by exact bbox, try a looser match: overlap
                        if not matched_block:
                            for db in dict_blocks:
                                if db.get("type") != 0:
                                    continue
                                db_bbox = db.get("bbox") or []
                                if len(db_bbox) >= 4:
                                    dx0, dy0, dx1, dy1 = db_bbox
                                    # Check center of block lies within
                                    cx = (x0 + x1) / 2
                                    cy = (y0 + y1) / 2
                                    if (
                                        dx0 - tolerance <= cx <= dx1 + tolerance
                                        and dy0 - tolerance <= cy <= dy1 + tolerance
                                    ):
                                        matched_block = db
                                        break

                        if matched_block:
                            # gather fonts, sizes, colors from spans
                            spans = []
                            for line in matched_block.get("lines", []):
                                for span in line.get("spans", []):
                                    spans.append(span)

                            if spans:
                                # Get most frequent font and size, and a common color
                                fonts = [s.get("font") or "" for s in spans]
                                sizes = [s.get("size") for s in spans if s.get("size")]
                                colors = [
                                    s.get("color") for s in spans if s.get("color")
                                ]

                                # pick most frequent non-empty font
                                from collections import Counter

                                try:
                                    font_name = Counter(
                                        [f for f in fonts if f]
                                    ).most_common(1)[0][0]
                                except Exception:
                                    font_name = ""

                                try:
                                    # pick typical size (rounded)
                                    size_counts = Counter(
                                        [round(float(s), 1) for s in sizes]
                                    )
                                    font_size = (
                                        ""
                                        if not size_counts
                                        else f"{size_counts.most_common(1)[0][0]:.1f}"
                                    )
                                except Exception:
                                    font_size = ""

                                try:
                                    # Normalize color values (PyMuPDF can give integer or tuples)
                                    def _col_to_hex(c):
                                        try:
                                            if c is None:
                                                return ""
                                            # integer colors (24-bit) -- convert to RGB
                                            if isinstance(c, int):
                                                r = (c >> 16) & 255
                                                g = (c >> 8) & 255
                                                b = c & 255
                                                return f"#{r:02x}{g:02x}{b:02x}"
                                            # tuple of floats 0..1 or ints 0..255
                                            if (
                                                isinstance(c, (tuple, list))
                                                and len(c) >= 3
                                            ):
                                                r, g, b = c[0], c[1], c[2]
                                                if isinstance(r, float) and r <= 1.0:
                                                    r = int(round(r * 255))
                                                    g = int(round(g * 255))
                                                    b = int(round(b * 255))
                                                else:
                                                    r, g, b = int(r), int(g), int(b)
                                                return f"#{r:02x}{g:02x}{b:02x}"
                                            # string colors (like "#rrggbb")
                                            if isinstance(c, str) and c.startswith("#"):
                                                return c
                                        except Exception:
                                            return ""
                                        return ""

                                    # pick most common color hex
                                    hex_colors = [_col_to_hex(c) for c in colors]
                                    hex_counts = Counter([h for h in hex_colors if h])
                                    font_color = (
                                        hex_counts.most_common(1)[0][0]
                                        if hex_counts
                                        else ""
                                    )
                                except Exception:
                                    font_color = ""

                except Exception:
                    font_name = ""
                    font_size = ""
                    font_color = ""

                tb.set("font", font_name)
                tb.set("size", font_size)
                tb.set("color", font_color)
                # classify type heuristically using font size if available
                try:
                    parsed_size = float(tb.get("size")) if tb.get("size") else 0.0
                except Exception:
                    parsed_size = 0.0
                ttype = self._classify_text_type(text, parsed_size, (x0, y0, x1, y1))
                tb.set("type", ttype)
                tb.text = text.strip()
                page_elem.append(tb)

        except Exception as e:
            print(f"⚠️ Failed to extract text blocks on page {page_num+1}: {e}")

        # Image extraction - FIXED VERSION
        try:
            debug_images_env = os.getenv("PDF_EXTRACTOR_DEBUG_IMAGES", "")
            debug_images = debug_images_env in ("1", "true", "yes", "y")

            # Get page boundaries for validation
            page_rect = page.rect

            # Get images that are actually ON this page
            image_list = page.get_images(full=True)

            # Tuning knobs via environment variables (with sensible defaults)
            min_side_pts = float(os.getenv("PDF_EXTRACTOR_MIN_RECT_SIDE_PTS", "25"))
            min_area_frac = float(
                os.getenv("PDF_EXTRACTOR_MIN_RECT_AREA_FRAC", "0.001")
            )
            max_occ_per_xref = int(
                os.getenv("PDF_EXTRACTOR_MAX_OCCURRENCES_PER_IMAGE", "1")
            )

            # Per-page visual dedup using simple perceptual hash (dhash)
            seen_hashes = set()

            # Helper: compute IoU between two rectangles
            def _rect_iou(r1: fitz.Rect, r2: fitz.Rect) -> float:
                ix0 = max(r1.x0, r2.x0)
                iy0 = max(r1.y0, r2.y0)
                ix1 = min(r1.x1, r2.x1)
                iy1 = min(r1.y1, r2.y1)
                iw = max(0.0, ix1 - ix0)
                ih = max(0.0, iy1 - iy0)
                inter = iw * ih
                a1 = max(0.0, r1.width) * max(0.0, r1.height)
                a2 = max(0.0, r2.width) * max(0.0, r2.height)
                union = a1 + a2 - inter if (a1 + a2 - inter) > 0 else 0.0
                return inter / union if union > 0 else 0.0

        except Exception:
            image_list = []

        # Track processed xrefs to avoid duplicates
        processed_xrefs = set()
        # Track seen bboxes to avoid repeating identical failed entries
        seen_bboxes = set()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]

                # Skip if we've already processed this xref on this page
                if xref in processed_xrefs:
                    continue

                # Get all rectangles where this image appears
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                # Filter rectangles to only include those that are actually within the page bounds
                # and have sufficient size (absolute min side and page-relative area)
                valid_rects = []
                page_area = (
                    page_rect.width * page_rect.height
                    if page_rect.width and page_rect.height
                    else 0
                )
                for rect in img_rects:
                    # Check if rectangle is within page boundaries with some tolerance
                    tolerance = 1.0  # PDF points
                    if not (
                        rect.x0 >= page_rect.x0 - tolerance
                        and rect.y0 >= page_rect.y0 - tolerance
                        and rect.x1 <= page_rect.x1 + tolerance
                        and rect.y1 <= page_rect.y1 + tolerance
                    ):
                        continue
                    # Absolute min side in points
                    if rect.width < min_side_pts or rect.height < min_side_pts:
                        continue
                    # Page-relative area threshold
                    if page_area > 0 and (rect.width * rect.height) < (
                        page_area * min_area_frac
                    ):
                        continue
                    valid_rects.append(rect)

                # Skip if no valid rectangles found on this page
                if not valid_rects:
                    continue

                # Mark this xref as processed
                processed_xrefs.add(xref)

                # Reduce overlaps using a simple non-maximum suppression; keep largest areas first
                valid_rects.sort(key=lambda r: (r.width * r.height), reverse=True)
                nms_kept = []
                for r in valid_rects:
                    if all(_rect_iou(r, k) < 0.5 for k in nms_kept):
                        nms_kept.append(r)
                    if len(nms_kept) >= max_occ_per_xref:
                        break

                # Process each selected rectangle occurrence
                for r_idx, rect in enumerate(nms_kept):

                    try:
                        # Create unique image filename for this occurrence
                        image_filename = os.path.join(
                            image_folder,
                            f"page_{page_num+1}_image_{img_index}_{r_idx}.png",
                        )

                        # Extract image using page clipping (more accurate)
                        pad = 0.5  # Small padding for anti-aliasing
                        clip_rect = fitz.Rect(
                            max(rect.x0 - pad, page_rect.x0),
                            max(rect.y0 - pad, page_rect.y0),
                            min(rect.x1 + pad, page_rect.x1),
                            min(rect.y1 + pad, page_rect.y1),
                        )

                        # Render at higher resolution for better quality
                        zoom = 2  # 2x resolution
                        mat = fitz.Matrix(zoom, zoom)

                        extraction_success = False
                        try:
                            # Extract by clipping the page (most accurate method)
                            clipped_pix = page.get_pixmap(
                                matrix=mat, clip=clip_rect, alpha=False
                            )
                            clipped_pix.save(image_filename)
                            extraction_success = True
                            clipped_pix = None  # Free memory
                        except Exception:
                            # Fallback: try to extract the raw image object
                            try:
                                pix = fitz.Pixmap(page.parent, xref)
                                if pix.n < 5:
                                    pix.save(image_filename)
                                else:
                                    conv = fitz.Pixmap(fitz.csRGB, pix)
                                    conv.save(image_filename)
                                    conv = None
                                extraction_success = True
                                pix = None  # Free memory
                            except Exception:
                                extraction_success = False

                        # Simple color shade filter: skip images with very few unique colors (likely text)
                        if extraction_success:
                            try:
                                from PIL import Image as PILImage
                                import numpy as np

                                with PILImage.open(image_filename) as img:
                                    arr = np.array(img.convert("RGB"))
                                    pixels = arr.reshape(-1, arr.shape[2])
                                    unique_colors = len(np.unique(pixels, axis=0))
                                    print(
                                        f"Image {os.path.basename(image_filename)} unique color count: {unique_colors}"
                                    )
                                    # If image has very few unique colors (<=3), skip (likely text)
                                    if unique_colors <= 3:
                                        print(
                                            f"   🗑️  Skipped {os.path.basename(image_filename)}: only {unique_colors} unique colors (likely text)"
                                        )
                                        extraction_success = False
                                        try:
                                            os.remove(image_filename)
                                        except Exception:
                                            pass
                            except Exception as e:
                                print(
                                    f"   ⚠️  Color count check failed for {os.path.basename(image_filename)}: {e}"
                                )

                        # Visual dedup across the page using a lightweight dhash
                        if extraction_success:
                            try:
                                from PIL import Image as _PIL
                                import numpy as _np

                                with _PIL.open(image_filename) as _im:
                                    g = _im.convert("L").resize(
                                        (9, 8), _PIL.Resampling.LANCZOS
                                    )
                                    garr = _np.array(g)
                                    diff = garr[:, 1:] > garr[:, :-1]
                                    bits = "".join(
                                        "1" if v else "0" for v in diff.flatten()
                                    )
                                    dhash_hex = f"{int(bits, 2):0{len(bits)//4}x}"
                                if dhash_hex in seen_hashes:
                                    # Duplicate visual content; remove file and skip
                                    try:
                                        os.remove(image_filename)
                                    except Exception:
                                        pass
                                    extraction_success = False
                                else:
                                    seen_hashes.add(dhash_hex)
                            except Exception:
                                # Fail-open if hashing fails
                                pass

                        # If extraction succeeded, attempt to detect text-like/ghost images and optionally OCR them
                        if extraction_success:
                            try:
                                # If image looks like a ghost/text artifact, convert it to a text block
                                is_ghost = False
                                try:
                                    is_ghost = self._is_ghost_or_text_image(
                                        image_filename
                                    )
                                except Exception:
                                    is_ghost = False

                                if is_ghost:
                                    ocr_text = ""
                                    if self.ocr_available:
                                        try:
                                            with Image.open(image_filename) as _ocr_img:
                                                # Allow pytesseract to extract whatever text is present
                                                ocr_text = pytesseract.image_to_string(
                                                    _ocr_img
                                                )
                                        except Exception:
                                            ocr_text = ""

                                    # Remove the visual file since it's a text artifact
                                    try:
                                        os.remove(image_filename)
                                    except Exception:
                                        pass

                                    # If OCR found text, add it as a text_block into the XML
                                    if ocr_text and ocr_text.strip():
                                        tb = ET.Element("text_block")
                                        tb.set(
                                            "id",
                                            f"text_ocr_{page_num}_{img_index}_{r_idx}",
                                        )
                                        tb.set(
                                            "bbox",
                                            f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}",
                                        )
                                        try:
                                            tb.set("width", f"{(rect.x1-rect.x0):.1f}")
                                            tb.set("height", f"{(rect.y1-rect.y0):.1f}")
                                        except Exception:
                                            tb.set("width", "")
                                            tb.set("height", "")
                                        tb.set("font", "")
                                        tb.set("color", "")
                                        tb.set("size", "")
                                        tb.set("type", "ocr_text")
                                        tb.text = ocr_text.strip()
                                        page_elem.append(tb)
                                    else:
                                        # No OCR text found; mark as skipped (no image element)
                                        pass

                                    # Do not create an image_block for ghost/text images
                                    extraction_success = False
                                else:
                                    # Create XML element for a real image
                                    img_block = ET.Element("image_block")
                                    img_block.set(
                                        "id", f"img_{page_num}_{img_index}_{r_idx}"
                                    )
                                    img_block.set(
                                        "bbox",
                                        f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}",
                                    )
                                    img_block.set("width", f"{rect.width:.1f}")
                                    img_block.set("height", f"{rect.height:.1f}")
                                    img_block.set("file", image_filename)
                                    img_block.set(
                                        "xref", str(xref)
                                    )  # Add xref for debugging
                                    page_elem.append(img_block)
                            except Exception as e:
                                print(
                                    f"⚠️ Image post-processing failed for {image_filename}: {e}"
                                )
                                # Fallback to a failed marker (so downstream tooling knows something was here)
                                img_block = ET.Element("image_block")
                                img_block.set(
                                    "id", f"img_{page_num}_{img_index}_{r_idx}"
                                )
                                img_block.set(
                                    "bbox",
                                    f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}",
                                )
                                img_block.set("width", f"{rect.width:.1f}")
                                img_block.set("height", f"{rect.height:.1f}")
                                img_block.set("file", "extraction_failed")
                                img_block.set("xref", str(xref))
                                page_elem.append(img_block)
                        else:
                            # Try fallback: crop the saved full-page image and attempt ghost detection / OCR
                            bbox_str = f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}"
                            # Avoid repeating the same failed bbox many times
                            if bbox_str in seen_bboxes:
                                continue
                            seen_bboxes.add(bbox_str)

                            try:
                                full_page_img_path = os.path.join(
                                    image_folder, f"page_{page_num+1}_full.png"
                                )
                                fallback_text = ""
                                fallback_detected = False
                                if os.path.exists(full_page_img_path):
                                    try:
                                        from PIL import Image as _PIL

                                        with _PIL.open(full_page_img_path) as _full:
                                            fw, fh = _full.size
                                            # Compute scale from PDF points to full image pixels
                                            try:
                                                scale_x = fw / (
                                                    page_rect.width
                                                    if page_rect.width
                                                    else 1
                                                )
                                                scale_y = fh / (
                                                    page_rect.height
                                                    if page_rect.height
                                                    else 1
                                                )
                                            except Exception:
                                                scale_x = scale_y = 2

                                            crop_box = (
                                                int(max(0, rect.x0 * scale_x)),
                                                int(max(0, rect.y0 * scale_y)),
                                                int(min(fw, rect.x1 * scale_x)),
                                                int(min(fh, rect.y1 * scale_y)),
                                            )
                                            if (
                                                crop_box[2] > crop_box[0]
                                                and crop_box[3] > crop_box[1]
                                            ):
                                                crop = _full.crop(crop_box)
                                                tmp_crop_path = os.path.join(
                                                    image_folder,
                                                    f"page_{page_num+1}_crop_fallback_{img_index}_{r_idx}.png",
                                                )
                                                crop.save(tmp_crop_path)
                                                # Run ghost detection on the crop
                                                try:
                                                    if self._is_ghost_or_text_image(
                                                        tmp_crop_path
                                                    ):
                                                        fallback_detected = True
                                                except Exception:
                                                    fallback_detected = False

                                                # If detected as ghost/text and OCR available, try to OCR
                                                if (
                                                    fallback_detected
                                                    and self.ocr_available
                                                ):
                                                    try:
                                                        with Image.open(
                                                            tmp_crop_path
                                                        ) as _ocr_img:
                                                            fallback_text = pytesseract.image_to_string(
                                                                _ocr_img
                                                            )
                                                    except Exception:
                                                        fallback_text = ""
                                                # Remove the temporary crop file
                                                try:
                                                    os.remove(tmp_crop_path)
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass

                                # If fallback OCR found text, append as text_block
                                if fallback_text and fallback_text.strip():
                                    tb = ET.Element("text_block")
                                    tb.set(
                                        "id",
                                        f"text_ocr_fallback_{page_num}_{img_index}_{r_idx}",
                                    )
                                    tb.set(
                                        "bbox",
                                        bbox_str,
                                    )
                                    try:
                                        # compute width/height from bbox string
                                        bx0, by0, bx1, by1 = [
                                            float(x) for x in bbox_str.split(",")
                                        ]
                                        tb.set("width", f"{(bx1-bx0):.1f}")
                                        tb.set("height", f"{(by1-by0):.1f}")
                                    except Exception:
                                        tb.set("width", "")
                                        tb.set("height", "")
                                    tb.set("font", "")
                                    tb.set("color", "")
                                    tb.set("size", "")
                                    tb.set("type", "ocr_text")
                                    tb.text = fallback_text.strip()
                                    page_elem.append(tb)
                                    continue
                            except Exception as e:
                                print(
                                    f"     ⚠️ Fallback crop/OCR failed for bbox {bbox_str}: {e}"
                                )

                            # If fallback didn't yield text, add a single extraction_failed marker for this bbox
                            img_block = ET.Element("image_block")
                            img_block.set("id", f"img_{page_num}_{img_index}_{r_idx}")
                            img_block.set(
                                "bbox",
                                bbox_str,
                            )
                            img_block.set("width", f"{rect.width:.1f}")
                            img_block.set("height", f"{rect.height:.1f}")
                            img_block.set("file", "extraction_failed")
                            img_block.set("xref", str(xref))
                            page_elem.append(img_block)

                    except Exception as e:
                        print(
                            f"⚠️ Failed to process image {img_index}_{r_idx} on page {page_num+1}: {e}"
                        )
                        continue

            except Exception as e:
                print(
                    f"⚠️ Failed to process image {img_index} on page {page_num+1}: {e}"
                )
                continue

        return page_elem

    def _classify_text_type(self, text, font_size, bbox):
        """Classify text as title, description, price, etc."""
        text_upper = text.upper()

        # Large text is likely a title
        if font_size > 16:
            return "title"

        # Check for product codes (like "2,5AL - UBER")
        if re.search(r"\d+[,.]?\d*\s*[A-Z]{1,3}", text):
            return "product_code"

        # Check for prices (€, $, £, etc.)
        if re.search(r"[€$£¥₹]\s*\d+|^\d+[.,]\d*\s*[€$£¥₹]", text):
            return "price"

        # Short descriptive phrases
        if len(text.split()) <= 5 and any(
            word in text_upper
            for word in ["CLASSIC", "MODERN", "LUXURY", "PREMIUM", "STYLE"]
        ):
            return "style_tag"

        # Longer text is description
        if len(text) > 50:
            return "description"

        return "text"

    def _save_pretty_xml(self, root, output_path):
        """Save XML with pretty formatting, sanitize text"""
        import xml.sax.saxutils as saxutils
        import string

        def clean_text(text):
            """Remove invalid XML characters and control characters"""
            if not text:
                return text

            # Keep only valid XML characters
            # Valid XML 1.0 characters: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
            valid_chars = []
            for char in text:
                code_point = ord(char)
                if (
                    code_point == 0x09  # Tab
                    or code_point == 0x0A  # Line feed
                    or code_point == 0x0D  # Carriage return
                    or (0x20 <= code_point <= 0xD7FF)  # Basic multilingual plane
                    or (0xE000 <= code_point <= 0xFFFD)  # Private use area
                    or (0x10000 <= code_point <= 0x10FFFF)
                ):  # Supplementary planes
                    valid_chars.append(char)

            cleaned_text = "".join(valid_chars)
            # Also escape XML entities
            return saxutils.escape(cleaned_text)

        def sanitize_element(elem):
            if elem.text:
                elem.text = clean_text(elem.text)
            if elem.tail:
                elem.tail = clean_text(elem.tail)
            for attr_name, attr_value in elem.attrib.items():
                elem.set(attr_name, clean_text(attr_value))
            for child in elem:
                sanitize_element(child)

        sanitize_element(root)
        rough_string = ET.tostring(root, "unicode")
        try:
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
        except Exception as e:
            print(f"Warning: Could not pretty print XML: {e}")
            # Fallback: write rough XML if pretty fails
            pretty_xml = rough_string
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    def xml_to_json(self, xml_path: str, out_path: str = None) -> str:
        """Convert an extractor XML file to a simple JSON structure.

        Returns the path to the written JSON file.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise

        data = {}
        data["source"] = root.get("source") or xml_path
        data["total_pages"] = int(root.get("total_pages", 0))
        pages = []

        for p in root.findall("page"):
            try:
                page_number = int(p.get("number") or 0)
            except Exception:
                page_number = 0
            page_obj = {
                "page_number": page_number,
                "width": float(p.get("width") or 0),
                "height": float(p.get("height") or 0),
                "images": [],
                "texts": [],
            }
            # Blocking heuristics thresholds
            try:
                image_min_frac = float(os.getenv("PDF_IMAGE_MIN_AREA_FRAC", "0"))
                image_max_frac = float(os.getenv("PDF_IMAGE_MAX_AREA_FRAC", "0.45"))
                # New: thresholds for width/height proportional blocking
                image_min_dim_frac = float(os.getenv("PDF_IMAGE_MIN_DIM_FRAC", "0"))
                image_max_dim_frac = float(os.getenv("PDF_IMAGE_MAX_DIM_FRAC", "0.45"))
            except Exception:
                image_min_frac = 0
                image_max_frac = 0.45
                image_min_dim_frac = 0
                image_max_dim_frac = 0.45
            # attach a full-page snapshot path if it exists (images/page_{n}_full.png)
            try:
                xml_dir = os.path.dirname(xml_path) or "."
                candidate = os.path.join(
                    xml_dir, "images", f"page_{page_number}_full.png"
                )
                if os.path.exists(candidate):
                    page_obj["full_image"] = candidate
            except Exception:
                pass

            for img in p.findall("image_block"):
                bbox_attr = img.get("bbox") or ""
                bbox = (
                    [float(x) for x in bbox_attr.split(",") if x.strip()]
                    if bbox_attr
                    else []
                )
                page_obj["images"].append(
                    {
                        "id": img.get("id"),
                        "bbox": bbox,
                        "width": float(img.get("width")) if img.get("width") else None,
                        "height": (
                            float(img.get("height")) if img.get("height") else None
                        ),
                        "file": img.get("file"),
                        "xref": img.get("xref"),
                    }
                )
                # compute area fraction and optionally block too-large or too-small images
                try:
                    w = (
                        float(img.get("width"))
                        if img.get("width")
                        else (bbox[2] - bbox[0])
                    )
                    h = (
                        float(img.get("height"))
                        if img.get("height")
                        else (bbox[3] - bbox[1])
                    )
                    page_width = page_obj.get("width", 0) or 1
                    page_height = page_obj.get("height", 0) or 1
                    w_frac = w / page_width
                    h_frac = h / page_height
                    # NOTE: image filtering removed — keep all image blocks so grouping
                    # can consider them as anchors. Previously we flagged images as
                    # blocked according to dimension/area thresholds which caused
                    # downstream grouping to skip them and leave lone text groups.
                    # To satisfy the user's request, stop marking images blocked
                    # here and keep a clean, consistent structure.
                    page_obj["images"][-1]["blocked"] = False
                except Exception:
                    page_obj["images"][-1]["blocked"] = False

            for tb in p.findall("text_block"):
                bbox_attr = tb.get("bbox") or ""
                bbox = (
                    [float(x) for x in bbox_attr.split(",") if x.strip()]
                    if bbox_attr
                    else []
                )
                page_obj["texts"].append(
                    {
                        "id": tb.get("id"),
                        "content": tb.text or "",
                        "type": tb.get("type"),
                        "bbox": bbox,
                        "font": tb.get("font"),
                        "size": (
                            float(tb.get("size"))
                            if tb.get("size") and tb.get("size").strip()
                            else None
                        ),
                        "color": tb.get("color"),
                    }
                )
                # Block blocks that are numeric-only. We define numeric-only as
                # a string composed entirely of one or more numeric tokens
                # (integers or decimals) possibly separated by whitespace and
                # common separators such as comma, dot, plus, minus, colon or slash.
                # Examples that will be blocked: '14', '137\n138', '1,234', '3.14', '+100 -200'
                try:
                    content = (tb.text or "").strip()
                    if content:
                        # Match sequences of numeric tokens separated by whitespace or
                        # simple punctuation. This rejects tokens containing letters
                        # (eg '400x100'). \n is included in \s so multiline numeric
                        # blocks like '137\n138' will be considered numeric-only.
                        if re.match(
                            r"^[\s\+\-]*\d+(?:[\.,]\d+)?(?:[\s,;:/\-\+]+\d+(?:[\.,]\d+)?)*\s*$",
                            content,
                        ):
                            page_obj["texts"][-1]["blocked"] = True
                            page_obj["texts"][-1]["blocked_reason"] = "only_number"
                        else:
                            # Block single-character or short gibberish/product-code-like
                            # tokens to avoid confusing downstream AI/context.
                            # Configurable via PDF_BLOCK_SHORT_TOKENS (default true) and
                            # PDF_BLOCK_SHORT_TOKENS_MAX_LEN controls token max length considered (default 3).
                            try:
                                if os.getenv(
                                    "PDF_BLOCK_SHORT_TOKENS", "1"
                                ).strip() not in (
                                    "0",
                                    "false",
                                    "no",
                                ):
                                    max_len = int(
                                        os.getenv("PDF_BLOCK_SHORT_TOKENS_MAX_LEN", "3")
                                    )
                                else:
                                    max_len = 0
                            except Exception:
                                max_len = 3

                            if max_len and content:
                                # Split into tokens (whitespace). If single token or all tokens are 1-char,
                                # apply heuristics.
                                toks = re.split(r"\s+", content)
                                # mark single-character or single-token codes as blocked
                                if len(toks) == 1:
                                    tok = toks[0]
                                    tlen = len(tok)
                                    if tlen == 1:
                                        page_obj["texts"][-1]["blocked"] = True
                                        page_obj["texts"][-1][
                                            "blocked_reason"
                                        ] = "single_char"
                                    else:
                                        # heuristics for short noisy tokens (product codes / gibberish)
                                        # If token is short (<= max_len) and contains digits or
                                        # non-alpha characters, or is mostly non-alpha, treat as noise.
                                        alpha_count = sum(1 for c in tok if c.isalpha())
                                        digit_count = sum(1 for c in tok if c.isdigit())
                                        non_alnum = sum(
                                            1 for c in tok if not c.isalnum()
                                        )
                                        alpha_ratio = alpha_count / float(max(1, tlen))
                                        if tlen <= max_len and (
                                            digit_count > 0
                                            or non_alnum > 0
                                            or alpha_ratio < 0.5
                                        ):
                                            page_obj["texts"][-1]["blocked"] = True
                                            page_obj["texts"][-1][
                                                "blocked_reason"
                                            ] = "short_code_or_gibberish"
                                else:
                                    # multi-token content: if every token is a single-char token,
                                    # block it (e.g., 'A B C'). Also if there is one token and others are punctuation
                                    all_one_char = all(
                                        len(t) == 1 for t in toks if t.strip()
                                    )
                                    if all_one_char and len(toks) > 0:
                                        page_obj["texts"][-1]["blocked"] = True
                                        page_obj["texts"][-1][
                                            "blocked_reason"
                                        ] = "single_char_run"
                except Exception:
                    # best effort; don't break extraction on regex failures
                    pass
                # store size value on page object (already set in tb element)
                # we'll apply per-page density filtering after collecting all
                # text blocks for this page so blocking is based on page-local
                # font-density instead of a static global min/max.
                try:
                    parsed_size = (
                        float(tb.get("size"))
                        if tb.get("size") and tb.get("size").strip()
                        else None
                    )
                except Exception:
                    parsed_size = None
                # keep the float size value in the JSON so downstream steps
                # can use it if needed
                try:
                    page_obj["texts"][-1]["size"] = parsed_size
                except Exception:
                    pass

            # --- PER-PAGE font-density based blocking ---
            # If any text blocks lack a numeric size, mark them as blocked to avoid
            # downstream grouping ambiguity. However, if a large fraction of the
            # page's text blocks have missing sizes (extraction failure for that
            # page), we should NOT block them all — treat that as an extraction
            # artifact. The skip ratio is configurable via
            # `PDF_TEXT_MISSING_SIZE_SKIP_RATIO` (default 0.5 = 50%).
            try:
                skip_ratio = float(os.getenv("PDF_TEXT_MISSING_SIZE_SKIP_RATIO", "0.2"))
            except Exception:
                skip_ratio = 0.5

            texts_on_page = page_obj.get("texts", []) or []
            total_texts = len(texts_on_page)
            missing_count = sum(1 for t in texts_on_page if t.get("size") is None)
            # If the fraction of missing sizes is >= skip_ratio, assume page-wide
            # extraction issue and do not mass-block missing-size blocks.
            block_missing = True
            try:
                if (
                    total_texts > 0
                    and (missing_count / float(total_texts)) >= skip_ratio
                ):
                    block_missing = False
            except Exception:
                block_missing = True

            if block_missing:
                for t in texts_on_page:
                    try:
                        if t.get("size") is None:
                            if not t.get("blocked"):
                                t["blocked"] = True
                                t["blocked_reason"] = (
                                    t.get("blocked_reason") or "font_missing"
                                )
                    except Exception:
                        # best-effort: don't fail the page extraction if this step errors
                        pass
            # Collect sizes for the page and compute quantiles. Default to
            # sensible quantiles (5% low, 95% high) but allow overrides via
            # environment variables PDF_TEXT_PAGE_MIN_QUANTILE and PDF_TEXT_PAGE_MAX_QUANTILE
            try:
                q_low = float(os.getenv("PDF_TEXT_PAGE_MIN_QUANTILE", "0.05"))
                q_high = float(os.getenv("PDF_TEXT_PAGE_MAX_QUANTILE", "0.95"))
            except Exception:
                q_low = 0.05
                q_high = 0.95

            # gather numeric sizes present on the page
            sizes = [
                t["size"]
                for t in page_obj.get("texts", [])
                if t.get("size") is not None
            ]

            if sizes:
                # Normalize sizes by rounding decimals for consistent grouping
                try:
                    round_decimals = int(os.getenv("PDF_TEXT_SIZE_ROUND_DECIMALS", "0"))
                except Exception:
                    round_decimals = 0

                def key_size_local(v):
                    try:
                        return round(float(v), round_decimals)
                    except Exception:
                        return None

                # compute simple quantiles WITHOUT numpy on normalized (rounded) sizes
                sizes_norm = [
                    key_size_local(s) for s in sizes if key_size_local(s) is not None
                ]
                sizes_sorted = sorted(sizes_norm)
                n = len(sizes_sorted)

                def pick_quantile(arr, q):
                    if not arr:
                        return None
                    idx = q * (len(arr) - 1)
                    lo = int(idx)
                    hi = min(lo + 1, len(arr) - 1)
                    frac = idx - lo
                    return arr[lo] * (1 - frac) + arr[hi] * frac

                # Quantiles are computed on the rounded-size values so that the
                # outlier decision is consistent per-rounded-key across the page.
                page_min_size = pick_quantile(sizes_sorted, max(0.0, min(1.0, q_low)))
                page_max_size = pick_quantile(sizes_sorted, max(0.0, min(1.0, q_high)))

                # If computed quantiles collapse (page has mostly same size),
                # expand a small margin to avoid over-blocking; this uses a
                # small relative epsilon.
                if (
                    page_min_size is not None
                    and page_max_size is not None
                    and page_max_size - page_min_size < 1e-6
                ):
                    eps = max(0.5, page_min_size * 0.05)
                    page_min_size = max(0.0, page_min_size - eps)
                    page_max_size = page_max_size + eps

                # apply page-local blocking; keep any prior blocked flag (e.g., only_number)
                # Additionally apply a rarity-based filter (font-size frequency per page)
                try:
                    keep_ratio = float(os.getenv("PDF_TEXT_SIZE_KEEP_RATIO", "0.15"))
                except Exception:
                    keep_ratio = 0.15

                # Only enforce rarity rule for pages with at least this many text blocks
                try:
                    min_texts_for_rarity = int(
                        os.getenv("PDF_TEXT_MIN_PAGE_TEXTS", "6")
                    )
                except Exception:
                    min_texts_for_rarity = 6

                # Round sizes for histogram counting: use decimals to normalize
                try:
                    round_decimals = int(os.getenv("PDF_TEXT_SIZE_ROUND_DECIMALS", "0"))
                except Exception:
                    round_decimals = 0

                # Treat missing/null sizes as their own unique bucket key so that
                # we can reason about their frequency/char-density independently.
                NULL_KEY = "__NULL__"

                def key_size(v):
                    # Return a consistent key for histogram/char buckets.
                    if v is None:
                        return NULL_KEY
                    try:
                        return round(float(v), round_decimals)
                    except Exception:
                        return NULL_KEY

                # Only consider text blocks with actual content when computing frequencies
                text_blocks_with_content = [
                    t for t in page_obj.get("texts", []) if t.get("content")
                ]
                total_texts = len(text_blocks_with_content)

                # build frequency map for rounded sizes (by block count)
                freq = {}
                # build char-count map for rounded sizes (by total characters)
                freq_chars = {}
                total_chars = 0
                for t in text_blocks_with_content:
                    s = t.get("size")
                    ks = key_size(s)
                    freq[ks] = freq.get(ks, 0) + 1
                    chars = len((t.get("content") or "").strip())
                    freq_chars[ks] = freq_chars.get(ks, 0) + chars
                    total_chars += chars

                # Determine rarity mode and thresholds. Modes: 'blocks', 'chars', 'both'
                rarity_mode = os.getenv("PDF_TEXT_SIZE_RARITY_MODE", "chars").lower()
                try:
                    keep_ratio_block = float(
                        os.getenv("PDF_TEXT_SIZE_KEEP_RATIO", "0.15")
                    )
                except Exception:
                    keep_ratio_block = 0.15
                try:
                    keep_ratio_char = float(
                        os.getenv("PDF_TEXT_SIZE_KEEP_RATIO_CHAR", "0.10")
                    )
                except Exception:
                    keep_ratio_char = 0.10
                try:
                    min_chars_for_rarity = int(
                        os.getenv("PDF_TEXT_SIZE_MIN_CHARS", "30")
                    )
                except Exception:
                    min_chars_for_rarity = 30

                # Decide which rounded-size keys are considered "rare" according to the selected mode.
                rare_keys = set()
                if total_texts >= min_texts_for_rarity or (
                    rarity_mode == "chars" and total_chars >= min_chars_for_rarity
                ):
                    # Evaluate each ks observed on the page
                    for ks in set(list(freq.keys()) + list(freq_chars.keys())):
                        block_ratio = (
                            freq.get(ks, 0) / float(total_texts)
                            if total_texts > 0
                            else 0.0
                        )
                        char_ratio = (
                            freq_chars.get(ks, 0) / float(total_chars)
                            if total_chars > 0
                            else 0.0
                        )

                        is_rare = False
                        if rarity_mode == "blocks":
                            is_rare = block_ratio < keep_ratio_block
                        elif rarity_mode == "chars":
                            # require enough chars to make a decision
                            if total_chars >= min_chars_for_rarity:
                                is_rare = char_ratio < keep_ratio_char
                        else:  # both
                            # mark rare only if BOTH block and char ratios are below thresholds
                            if (
                                total_chars >= min_chars_for_rarity
                                and total_texts >= min_texts_for_rarity
                            ):
                                is_rare = (block_ratio < keep_ratio_block) and (
                                    char_ratio < keep_ratio_char
                                )

                        if is_rare:
                            rare_keys.add(ks)

                # apply blocking: outlier (quantile) OR rarity (low frequency/char-density)
                # We do per-rounded-key all-or-none assignment for rarity keys to avoid mixed states.
                for t in page_obj.get("texts", []):
                    s = t.get("size")
                    # outlier checks (quantile computed earlier) - preserve prior reasons
                    if s is not None:
                        if page_min_size is not None and s < page_min_size:
                            if not t.get("blocked"):
                                t["blocked"] = True
                                t["blocked_reason"] = (
                                    t.get("blocked_reason") or "font_too_small"
                                )
                            continue
                        if page_max_size is not None and s > page_max_size:
                            if not t.get("blocked"):
                                t["blocked"] = True
                                t["blocked_reason"] = (
                                    t.get("blocked_reason") or "font_too_large"
                                )
                            continue

                    # rarity check: apply per-rounded-key decision (includes NULL_KEY)
                    ks = key_size(s)
                    if ks in rare_keys:
                        # Only set blocked if no stronger reason already exists
                        if not t.get("blocked"):
                            t["blocked"] = True
                            t["blocked_reason"] = (
                                t.get("blocked_reason") or "font_too_rare"
                            )

            else:
                # If no per-page sizes found, fall back to static env values
                try:
                    text_min = float(os.getenv("PDF_TEXT_MIN_SIZE", "6"))
                    text_max = float(os.getenv("PDF_TEXT_MAX_SIZE", "12"))
                except Exception:
                    text_min = 6.0
                    text_max = 12.0
                for t in page_obj.get("texts", []):
                    s = t.get("size")
                    if s is None:
                        continue
                    if s < text_min:
                        t["blocked"] = True
                        t["blocked_reason"] = "font_too_small"
                    elif s > text_max:
                        t["blocked"] = True
                        t["blocked_reason"] = "font_too_large"

            pages.append(page_obj)

        data["pages"] = pages

        if not out_path:
            out_path = os.path.splitext(xml_path)[0] + ".json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path

    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for AI analysis"""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None

            # Check file size (keep under 20MB for best performance)
            file_size = os.path.getsize(image_path)
            if file_size > 20 * 1024 * 1024:  # 20MB
                print(
                    f"Image {image_path} too large ({file_size/1024/1024:.1f}MB), skipping"
                )
                return None

            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def analyze_page_with_ai(self, page_data, colors):
        """Use AI to intelligently match images with related text"""
        if not self.ai_model:
            print("⚠️  AI model not available for analysis")
            return None

        # Helper: ensure AI doesn't invent image filenames
        def _validate_ai_images(local_page, ai_result):
            """Remove any image paths from ai_result that are not present in local_page['images']."""
            try:
                valid_full = set()
                valid_base = set()
                for img in local_page.get("images", []):
                    f = img.get("file")
                    if not f:
                        continue
                    valid_full.add(f)
                    valid_base.add(os.path.basename(f))

                # ai_result could be a list of products
                if isinstance(ai_result, list):
                    for prod in ai_result:
                        if not isinstance(prod, dict):
                            continue
                        imgs = prod.get("images")
                        if not imgs:
                            prod["images"] = []
                            continue
                        filtered = []
                        for ip in imgs:
                            if not isinstance(ip, str):
                                continue
                            if ip in valid_full or os.path.basename(ip) in valid_base:
                                filtered.append(ip)
                            else:
                                # try matching by basename prefix (sometimes AI omits page folder)
                                b = os.path.basename(ip)
                                for vf in valid_full:
                                    if vf.endswith(b):
                                        filtered.append(vf)
                                        break
                        prod["images"] = filtered
                    return ai_result
                else:
                    return ai_result
            except Exception:
                return ai_result

        try:
            # Prepare the content with both text and images
            # Build a stronger prompt: explicitly list available image filenames and tell the model not to invent any
            available_images = [
                img.get("file")
                for img in page_data.get("images", [])
                if img.get("file")
            ]
            available_images_list = (
                "\n".join([f"- {p}" for p in available_images])
                or "(no extracted images)"
            )

            content_parts = [
                {
                    "type": "text",
                    "text": f"""You are an expert in analyzing PDF catalogue pages with advanced product identification and consolidation capabilities. You will receive:
                        1. A full page image showing the complete layout
                        2. XML data with all text blocks and their coordinates, plus extracted image information

                        Your task is to:
                        1. Analyze the full page image visually to understand the overall layout and product presentation
                        2. Identify distinct products/items on this page using intelligent consolidation logic
                        3. Group related images, text blocks, and variants that belong to the same product (different colors, sizes, angles, or detail shots should be considered as ONE product)
                        4. For each consolidated product, determine which text blocks and images belong to it
                        5. Extract comprehensive product information including specifications, features, materials, and dimensions
                        6. Create enhanced descriptions and organize data according to the specified schema
                        7. Return a structured array of products

                        INTELLIGENT PRODUCT CONSOLIDATION RULES:
                        - If multiple images show the same item in different colors, angles, or detail shots, treat as ONE product with multiple images
                        - If text mentions "available in colors" or shows color variants, consolidate into a single product entry
                        - If similar items have only minor variations (size, color, finish), group them as one product with variants
                        - Only create separate products if they are fundamentally different items (different models, completely different purposes)
                        - Look for product codes/SKUs - similar codes often indicate variants of the same product
                        - Consider spatial proximity and visual grouping in the layout

                        COLOR IDENTIFICATION RULES:
                        - Analyze product images visually to determine actual colors, NOT text content
                        - Use the predefined colors array below to find the MOST SIMILAR color match for each product
                        - Do NOT generate new hex color codes - only use colors from the provided array
                        - Do NOT extract color information from text that might be product codes, SKUs, or other identifiers
                        - Use AI vision to identify the dominant colors visible in the product images, then match to the closest color from the predefined list
                        - If no close match is found in the predefined colors, use the closest available option

                        PREDEFINED COLORS ARRAY (use these colors only):
                        {json.dumps(colors) if colors else "[]"}

                        IMPORTANT: 
                        - If you see no clear products on this page, return an empty array []
                        - Respond with ONLY raw JSON, not wrapped in ```json``` code blocks
                        - Don't include any unmatched items, just the products you can clearly identify
                        - Ensure price is always a decimal number, use 0.00 if no price found
                        - Generate meaningful slugs based on product names
                        - Extract rich content for features, dimensions, and specifications when available
                        - Populate color information comprehensively when color variants are shown, using visual analysis to match colors from the predefined array

                        Return JSON in this exact format matching the schema:
                        [
                        {{
                            "price": 0.00,
                            "name": "product_name_or_title",
                            "image1": "image_file_path_1",
                            "image2": "image_file_path_2", 
                            "image3": "image_file_path_3",
                            "image4": "image_file_path_4",
                            "image5": "image_file_path_5",
                            "product_colors": [
                            {{
                                "name": "color_name",
                                "code": "#hex_color_code",
                                "documentId": "color_document_id"
                            }}
                            ],
                            "slug": "product-name-slug",
                            "short_description": "Brief product description or key selling points",
                            "product_materials": "Materials used in construction/manufacturing",
                            "product_color_image": [
                            {{
                                "product_color": "color_name",
                                "product_color_image": "image_path_for_this_color",
                                "product_color_price": "price_for_this_color_variant",
                                "product_color_size": "size_if_applicable"
                            }}
                            ],
                            "product_features": "- [Extract actual features from product information]\n- [Use bullet points for each feature]\n- [Include benefits and unique selling points]",
                            "product_dimensions": "- [Extract actual measurements from product data]\n- [Include length, width, height as available]\n- [Add weight if mentioned]",
                            "product_specifications": "- [Extract actual materials and finishes]\n- [Include installation requirements if mentioned]\n- [Add warranty information if available]"
                        }}
                        ]

                        COLOR VARIANT UNIQUENESS:
                        - Each product must have UNIQUE color variants - never duplicate the same color name/code/documentId
                        - If you see multiple images of the same color, only include that color ONCE in product_colors
                        - Each color in product_colors must represent a visually distinct variant

                        RICH TEXT FORMAT REQUIREMENTS:
                        - Use proper Markdown formatting (NOT HTML) for product_features, product_dimensions, and product_specifications
                        - Use headers (##), bullet points (-), and structured lists for better readability
                        - Extract and include actual product information from the page content and images
                        - Use clear, descriptive labels for measurements and specifications
                        - Only include information that is actually available in the source material

                        FIELD REQUIREMENTS:
                        - price: Always include as decimal (0.00 if not found)
                        - name: Clear, descriptive product name
                        - image1-5: Include available image paths, leave null if fewer than 5 images
                        - product_colors: Array of UNIQUE color options with names, hex color codes and documentId from the predefined colors array only (no duplicates) 
                        - slug: URL-friendly version of product name (lowercase, hyphens)
                        - short_description: Concise summary of the product
                        - product_materials: Materials information when available
                        - product_color_image: Detailed color variant information
                        - product_features: Markdown formatted list of key features and benefits
                        - product_dimensions: Markdown formatted dimensions, measurements, and sizing information  
                        - product_specifications: Markdown formatted technical specifications and detailed product information

                        XML DATA FOR THIS PAGE:
                        {self.format_page_xml_for_ai(page_data)}""",
                }
            ]

            # Append a clear, strict instruction listing available images to prevent hallucination
            available_images_text = {
                "type": "text",
                "text": (
                    "Available image filenames (only these are valid - do NOT invent others):\n"
                    + available_images_list
                    + '\n\nIf no image filenames are listed above, set "images": [] for every product.'
                ),
            }
            content_parts.append(available_images_text)

            # Add up to 10 images to the content
            image_count = 0
            max_images = 1

            image_path = os.path.join(
                self.image_dir, f"page_{page_data['page_number']}_full.png"
            )
            if (
                image_path
                and image_path != "extraction_failed"
                and os.path.exists(image_path)
            ):
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
                    if image_base64:
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            }
                        )
                    image_count += 1
                    print(
                        f"     📷 Added image {image_count}: {os.path.basename(image_path)}"
                    )
                except Exception as e:
                    print(f"     ⚠️ Failed to encode image {image_path}: {e}")
            else:
                print(f"     ⚠️ Image not found or inaccessible: {image_path}")

            # Create message with text and images
            message = HumanMessage(content=content_parts)

            # Get AI response with retry logic
            import time

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.ai_model.invoke([message])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"     ⚠️ API call failed (attempt {attempt + 1}), retrying in 5 seconds..."
                        )
                        time.sleep(5)
                        continue
                    else:
                        print(
                            f"     ❌ API call failed after {max_retries} attempts: {e}"
                        )
                        return None

            # Parse JSON response using the reusable function
            parsed = parse_ai_json_response(response.content, "product analysis")
            if parsed is None:
                return None

            # Validate image paths inside the parsed result to avoid hallucinated image references
            parsed = _validate_ai_images(page_data, parsed)
            return parsed

        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return None

    def format_page_for_ai(self, page_data):
        """Format page data for AI analysis"""
        formatted = f"PAGE {page_data['page_number']} - Dimensions: {page_data['width']}x{page_data['height']}\n\n"

        formatted += "IMAGES:\n"
        for i, img in enumerate(page_data["images"]):
            formatted += f"Image {i+1}:\n"
            formatted += f"  - File: {img['file']}\n"
            formatted += f"  - Position: {img['bbox']}\n"
            formatted += f"  - Size: {img['width']}x{img['height']}\n\n"

        formatted += "TEXT BLOCKS:\n"
        for i, text in enumerate(page_data["texts"]):
            formatted += f"Text {i+1}:\n"
            formatted += f"  - Type: {text['type']}\n"
            formatted += f"  - Content: {text['content']}\n"
            formatted += f"  - Position: {text['bbox']}\n"
            formatted += (
                f"  - Font: {text.get('font', '')} (Size: {text.get('size', '')})\n"
            )
            if text.get("width") and text.get("height"):
                formatted += f"  - Box Size: {text.get('width')}x{text.get('height')}\n"
            if text.get("color"):
                formatted += f"  - Color: {text.get('color')}\n\n"
            else:
                formatted += "\n"
            if text.get("color"):
                formatted += f"  - Color: {text.get('color')}\n\n"
            else:
                formatted += "\n"

        return formatted

    def format_page_xml_for_ai(self, page_data):
        """Format page XML data for AI analysis"""
        formatted = f"PAGE {page_data['page_number']} - Dimensions: {page_data['width']}x{page_data['height']}\n\n"

        formatted += "EXTRACTED IMAGES (with coordinates):\n"
        for i, img in enumerate(page_data["images"]):
            if "full.png" not in img["file"]:  # Don't list the full page image
                formatted += f"Image {i+1}:\n"
                formatted += f"  - File: {img['file']}\n"
                formatted += f"  - Position: {img['bbox']}\n"
                formatted += f"  - Size: {img['width']}x{img['height']}\n\n"

        formatted += "TEXT BLOCKS (with coordinates and classification):\n"
        for i, text in enumerate(page_data["texts"]):
            formatted += f"Text {i+1}:\n"
            formatted += f"  - Type: {text['type']}\n"
            formatted += f"  - Content: \"{text['content']}\"\n"
            formatted += f"  - Position: {text['bbox']}\n"
            formatted += (
                f"  - Font: {text.get('font', '')} (Size: {text.get('size', '')})\n"
            )
            if text.get("width") and text.get("height"):
                formatted += f"  - Box Size: {text.get('width')}x{text.get('height')}\n"
            if text.get("color"):
                formatted += f"  - Color: {text.get('color')}\n\n"
            else:
                formatted += "\n"

        return formatted

    def ai_enhanced_extraction(self, xml_path, output_paths=None):
        """Extract products using AI-enhanced analysis - simplified approach"""
        if not self.ai_model:
            print("⚠️  AI model not available. Only XML is available.")
            return None

        print("🤖 Starting AI-enhanced extraction...")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        all_products = []  # Collect all products from all pages

        # Initialize progress files if output paths provided
        if output_paths:
            ai_json_path = output_paths["xml_path"].replace(
                "_structure.xml", "_ai_products.json"
            )
            progress_file = output_paths["xml_path"].replace(
                "_structure.xml", "_progress.json"
            )

            # Initialize progress tracking
            progress_data = {
                "total_pages": len(root.findall("page")),
                "completed_pages": 0,
                "last_updated": "",
                "total_products_found": 0,
                "products": [],
            }

        for page in root.findall("page"):
            page_num = page.get("number")
            print(f"   Analyzing page {page_num} with AI...")

            # Prepare page data
            page_data = {
                "page_number": int(page_num),
                "width": page.get("width"),
                "height": page.get("height"),
                "images": [],
                "texts": [],
            }

            # Collect images (including full page image)
            for img in page.findall("image_block"):
                page_data["images"].append(
                    {
                        "id": img.get("id"),
                        "file": img.get("file"),
                        "bbox": img.get("bbox"),
                        "width": img.get("width"),
                        "height": img.get("height"),
                    }
                )

            # Collect texts
            for text in page.findall("text_block"):
                page_data["texts"].append(
                    {
                        "id": text.get("id"),
                        "content": text.text or "",
                        "type": text.get("type"),
                        "bbox": text.get("bbox"),
                        "font": text.get("font"),
                        "size": text.get("size"),
                        "color": text.get("color"),
                        "width": text.get("width"),
                        "height": text.get("height"),
                    }
                )

            # Define your query
            query = """
                query ProductColors($pagination: PaginationArg) {
                    productColors(pagination: $pagination) {
                        code
                        name
                        documentId
                    }
                }"""

            # Variables to send with the query
            variables = {"pagination": {"limit": -1}}

            try:
                result = graphql_requester(query, variables)
                colors = result.get("productColors")
            except Exception as e:
                print(str(e))

            # Analyze with AI
            page_products = self.analyze_page_with_ai(page_data, colors)
            if page_products and isinstance(page_products, list):
                # Add page number to each product for reference
                for product in page_products:
                    product["page_number"] = int(page_num)

                all_products.extend(page_products)
                print(f"   ✅ Page {page_num}: Found {len(page_products)} products")

                # Save progress after each page if output paths provided
                if output_paths:
                    from datetime import datetime

                    # Update progress data
                    progress_data["products"] = all_products
                    progress_data["completed_pages"] = int(page_num)
                    progress_data["total_products_found"] = len(all_products)
                    progress_data["last_updated"] = datetime.now().isoformat()

                    # Save progress file
                    with open(progress_file, "w", encoding="utf-8") as f:
                        json.dump(progress_data, f, indent=2, ensure_ascii=False)

                    # Save current products
                    with open(ai_json_path, "w", encoding="utf-8") as f:
                        json.dump(all_products, f, indent=2, ensure_ascii=False)

                    print(
                        f"   ✅ Progress saved: {len(all_products)} total products so far"
                    )
            elif page_products is not None and len(page_products) == 0:
                print(f"   ℹ️  Page {page_num}: No products found (empty page)")
            else:
                print(f"   ❌ Page {page_num}: AI analysis failed")

            # Add rate limiting between pages to avoid overwhelming the API
            import time

            time.sleep(2)  # Wait 2 seconds between pages

        try:
            self.add_product_category(ai_json_path)
        except Exception as e:
            print(f"Error adding product category or sub category: {e}")

        print(f"\n🎉 AI extraction completed!")
        print(f"   📊 Total products found: {len(all_products)}")

        return all_products

    def _is_ghost_or_text_image(self, image_path: str) -> bool:
        """
        Detect if an image is likely a ghost/text artifact by analyzing multiple signals:
        - Color variance (low = monochrome/ghost)
        - Edge density (very sparse or very dense = text/noise)
        - White/black dominance
        - Histogram concentration
        - Saturation (grayscale-ish)
        - Extreme aspect ratios
        Decision uses a simple voting scheme to avoid false-positives on product images with white backgrounds.
        Returns True if image should be filtered out.
        """
        try:
            from PIL import Image as PILImage
            import numpy as np
        except ImportError:
            # If dependencies missing, skip ghost detection
            return False

        try:
            # Configure sensitivity via environment variable
            mode = os.getenv("PDF_EXTRACTOR_GHOST_MODE", "light").lower()
            dbg = os.getenv("PDF_EXTRACTOR_DEBUG_GHOST", "").lower() in (
                "1",
                "true",
                "yes",
                "y",
            )

            # Threshold presets per mode
            if mode == "strict":
                color_std_thresh = 25
                light_ratio_thresh = 0.70
                dark_ratio_thresh = 0.70
                edge_low, edge_high = 0.02, 0.30
                unique_colors_thresh = 8
                top10_ratio_thresh = 0.80
                saturation_thresh = 30
            elif mode == "balanced":
                color_std_thresh = 22
                light_ratio_thresh = 0.85
                dark_ratio_thresh = 0.85
                edge_low, edge_high = 0.015, 0.35
                unique_colors_thresh = 6
                top10_ratio_thresh = 0.85
                saturation_thresh = 25
            else:  # light (default)
                color_std_thresh = 15
                light_ratio_thresh = 0.92
                dark_ratio_thresh = 0.92
                edge_low, edge_high = 0.01, 0.45
                unique_colors_thresh = 5
                top10_ratio_thresh = 0.90
                saturation_thresh = 20

            with PILImage.open(image_path) as img:
                # Convert to RGB for consistent analysis
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Get original dimensions for aspect ratio check
                orig_width, orig_height = img.size
                aspect_ratio = max(orig_width, orig_height) / min(
                    orig_width, orig_height
                )

                # We'll collect all triggered reasons, then decide by mode (voting)
                reasons = []

                # Resize to reasonable analysis size (faster, still accurate)
                img.thumbnail((200, 200), PILImage.Resampling.LANCZOS)
                arr = np.array(img)

                # 1. Check color variance (ghost images are near-monochrome)
                # Calculate std deviation across RGB channels
                color_std = np.std(arr, axis=(0, 1))  # std per channel
                avg_color_std = np.mean(color_std)

                if avg_color_std < color_std_thresh:  # Low variance = flat/monochrome
                    reasons.append(
                        f"color_std {avg_color_std:.1f} < {color_std_thresh}"
                    )

                # 2. Check white/background dominance
                # Count pixels that are very light (near-white)
                light_threshold = 235  # RGB > 235 = very light
                light_pixels = np.all(arr > light_threshold, axis=2)
                light_ratio = np.sum(light_pixels) / (arr.shape[0] * arr.shape[1])

                if light_ratio > light_ratio_thresh:
                    reasons.append(
                        f"light_ratio {light_ratio:.2f} > {light_ratio_thresh}"
                    )

                # 2b. Check for near-black dominance (dark text on dark bg)
                dark_threshold = 30
                dark_pixels = np.all(arr < dark_threshold, axis=2)
                dark_ratio = np.sum(dark_pixels) / (arr.shape[0] * arr.shape[1])
                if dark_ratio > dark_ratio_thresh:
                    reasons.append(f"dark_ratio {dark_ratio:.2f} > {dark_ratio_thresh}")

                # 3. Check edge density (text has specific patterns)
                try:
                    import cv2

                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

                    if edge_density < edge_low:  # Almost no edges = blank/uniform
                        reasons.append(f"edge_density {edge_density:.3f} < {edge_low}")
                    if edge_density > edge_high:  # Too many edges = text/noise
                        reasons.append(f"edge_density {edge_density:.3f} > {edge_high}")

                except ImportError:
                    # OpenCV not available, skip edge check
                    pass

                # 4. Check for pure black/white binary image (scanned text)
                unique_colors = len(np.unique(arr.reshape(-1, arr.shape[2]), axis=0))
                if unique_colors < unique_colors_thresh:
                    reasons.append(
                        f"unique_colors {unique_colors} < {unique_colors_thresh}"
                    )

                # 5. Check histogram distribution - text images have bimodal distribution
                gray_arr = np.mean(arr, axis=2).astype(np.uint8)
                hist, _ = np.histogram(gray_arr, bins=256, range=(0, 256))
                # Normalize histogram
                hist = hist / (arr.shape[0] * arr.shape[1])

                # Check if most pixels are concentrated in very few intensity levels
                # Sort histogram and check if top 10 bins contain >80% of pixels
                sorted_hist = np.sort(hist)[::-1]
                top_10_ratio = np.sum(sorted_hist[:10])
                if top_10_ratio > top10_ratio_thresh:  # Too concentrated = likely text
                    reasons.append(
                        f"top10_ratio {top_10_ratio:.2f} > {top10_ratio_thresh}"
                    )

                # 6. Check for low saturation (grayscale-ish images are often text)
                hsv = img.convert("HSV")
                hsv_arr = np.array(hsv)
                saturation = hsv_arr[:, :, 1]
                avg_saturation = np.mean(saturation)

                # Low saturation = grayscale = likely text
                if avg_saturation < saturation_thresh:
                    reasons.append(
                        f"saturation {avg_saturation:.1f} < {saturation_thresh}"
                    )

                # 7. Extreme aspect ratios often indicate text snippets or UI elements
                if aspect_ratio > 8:
                    reasons.append(f"aspect_ratio {aspect_ratio:.1f} > 8")

                # Decide using a simple voting rule to reduce false-positives
                votes = len(reasons)
                if mode == "strict":
                    decision = votes >= 1
                elif mode == "balanced":
                    decision = votes >= 2
                else:  # light
                    decision = votes >= 2

                if dbg and decision:
                    print(
                        f"     ghost[{mode}] removing {os.path.basename(image_path)} due to: {', '.join(reasons)}"
                    )
                return decision

        except Exception as e:
            print(f"     ⚠️ Ghost detection failed for {image_path}: {e}")
            return False  # Conservative: don't filter if analysis fails

    def filter_small_images(
        self,
        image_folder: str,
        xml_path: str,
        min_dim: int = 30,
        remove_files: bool = True,
    ) -> set:
        """
        Scan `image_folder` for common image files, delete those where:
        - width < min_dim or height < min_dim (size filter)
        - OR detected as ghost/text artifact (ghost filter)
        and remove references to those files from the XML at `xml_path`.
        Returns a set of basenames that were removed (or would be removed).
        """
        removed_basenames = set()

        if not os.path.isdir(image_folder):
            return removed_basenames

        patterns = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.gif", "*.webp")

        # Determine if ghost/text filter should be enabled via env flag
        ghost_env = os.getenv("PDF_EXTRACTOR_GHOST_FILTER", "")
        enable_ghost_filter = ghost_env.lower() in ("1", "true", "yes", "y")

        # Skip ghost filtering on clearly large images (likely product photos)
        ghost_skip_min_pixels = int(
            os.getenv("PDF_EXTRACTOR_GHOST_SKIP_MIN_PIXELS", "80000")
        )
        ghost_skip_min_side = int(os.getenv("PDF_EXTRACTOR_GHOST_SKIP_MIN_SIDE", "300"))

        print(
            "   🔍 Applying image filters: size"
            + (
                " + ghost detection"
                if enable_ghost_filter
                else " (ghost detection OFF)"
            )
            + "..."
        )
        ghost_count = 0
        size_count = 0

        for pat in patterns:
            for path in glob.glob(os.path.join(image_folder, pat)):
                try:
                    # Get image dimensions
                    from PIL import Image as _PILImage

                    with _PILImage.open(path) as _im:
                        w, h = _im.size
                except Exception:
                    try:
                        import fitz

                        pix = fitz.Pixmap(path)
                        w, h = getattr(pix, "width", 0), getattr(pix, "height", 0)
                    except Exception:
                        continue

                should_remove = False
                removal_reason = ""

                basename = os.path.basename(path)
                is_full_page_image = "full" in basename

                # Always keep the full-page snapshot for analysis/debugging
                if is_full_page_image:
                    continue

                # Size filter
                if w < min_dim or h < min_dim:
                    should_remove = True
                    removal_reason = "size"
                    size_count += 1

                # Ghost/text filter (only check if size is OK and feature enabled)
                elif enable_ghost_filter:
                    # Protect large images from ghost removal
                    if (w * h) >= ghost_skip_min_pixels or max(
                        w, h
                    ) >= ghost_skip_min_side:
                        pass  # keep large images
                    elif self._is_ghost_or_text_image(path):
                        should_remove = True
                        removal_reason = "ghost"
                        ghost_count += 1

                if should_remove:
                    removed_basenames.add(os.path.basename(path))
                    if remove_files:
                        try:
                            os.remove(path)
                            # Optional: print each removal for debugging
                            # print(f"     🗑️  Removed ({removal_reason}): {os.path.basename(path)}")
                        except Exception:
                            pass

        print(
            f"   📊 Filtered totals: {size_count} too small, {ghost_count} ghost/text artifacts"
        )

        # Update XML: remove any <image_block> whose @file basename is in removed_basenames
        if os.path.exists(xml_path) and removed_basenames:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                changed = False
                # traverse all parents and remove direct children named image_block when matched
                for parent in list(root.iter()):
                    # findall returns direct children with that tag
                    for img in list(parent.findall("image_block")):
                        f = img.get("file", "")
                        if f and os.path.basename(f) in removed_basenames:
                            # capture tail text that would be preserved when removing the element
                            tail_text = img.tail or ""
                            # compute index of this child in parent's children list
                            children = list(parent)
                            try:
                                idx = children.index(img)
                            except ValueError:
                                idx = None

                            # remove the element
                            parent.remove(img)
                            changed = True

                            # Clean up small stray tails that often become isolated characters
                            tail_stripped = tail_text.strip()
                            if tail_stripped and len(tail_stripped) <= 3:
                                # discard very short tails (likely punctuation or artifact)
                                if idx is not None and idx > 0:
                                    prev = children[idx - 1]
                                    # remove any short tail on previous sibling
                                    prev.tail = (prev.tail or "").rstrip()
                                else:
                                    parent.text = (parent.text or "").rstrip()
                            else:
                                # if tail is significant, reattach it to previous sibling or parent
                                if tail_text:
                                    if idx is not None and idx > 0:
                                        prev = children[idx - 1]
                                        prev.tail = (prev.tail or "") + tail_text
                                    else:
                                        parent.text = (parent.text or "") + tail_text

                # If we removed anything, write the updated XML back to disk using the pretty writer
                if changed:
                    try:
                        # use the existing helper to sanitize and pretty-print
                        self._save_pretty_xml(root, xml_path)
                    except Exception:
                        # fallback to simple write if pretty save fails
                        try:
                            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                        except Exception:
                            pass
            except Exception:
                # if XML parsing fails, just return the removed set
                pass

        return removed_basenames

    def prepare_for_ai(
        self,
        xml_path: str,
        image_folder: str,
        min_dim: int = 30,
        remove_files: bool = True,
    ) -> set:
        """
        Convenience wrapper called before AI processing: filters small images and updates XML.
        Returns set of removed image basenames.
        """
        return self.filter_small_images(
            image_folder=image_folder,
            xml_path=xml_path,
            min_dim=min_dim,
            remove_files=remove_files,
        )

    def add_product_category(self, json_path: str, start_from=0):
        """Add product category to the JSON file"""
        categories = self.get_categories()

        with open(json_path, "r", encoding="utf-8") as f:
            products = json.load(f)

        total_products = len(products)
        print(f"Debug: start_from parameter = {start_from}")
        print(f"🏷️  Starting categorization for {total_products} products...")

        system_message = SystemMessage(
            content=f"""You are a product categorization AI. You will receive a product JSON object containing name, description, materials, features, and other details, along with an array of objects with name and documentId of categories. 

CRITICAL REQUIREMENTS:
- You MUST respond with ONLY valid JSON format
- Do NOT include any explanatory text, reasoning, or additional content
- Do NOT wrap your response in markdown code blocks
- Your response must be a single JSON object with "name" and "documentId" fields

TASK: Analyze the product's primary function, materials, and intended use to select the most appropriate category from the provided list. Prioritize the product's main purpose over secondary attributes. If multiple categories could apply, choose the most specific one.

RESPONSE FORMAT (return exactly this structure):
{{"name": "Selected Category Name", "documentId": "selected_category_document_id"}}

Available categories: {categories}

Remember: Respond with ONLY the JSON object, nothing else."""
        )

        for index, product in enumerate(products[start_from:], start_from + 1):
            try:
                print(
                    f"   📝 Processing product {index}/{total_products}: {product.get('name', 'Unknown')}"
                )

                human_message = HumanMessage(content=f"""Product: {product}""")

                # Get AI response with retry logic for category
                import time

                max_retries = 3
                category_response = None

                for attempt in range(max_retries):
                    try:
                        category_response = self.ai_model.invoke(
                            [system_message, human_message]
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(
                                f"     ⚠️ Category API call failed (attempt {attempt + 1}), retrying in 5 seconds..."
                            )
                            time.sleep(5)
                            continue
                        else:
                            print(
                                f"     ❌ Category API call failed after {max_retries} attempts: {e}"
                            )
                            raise e

                if category_response:
                    parsed_category = parse_ai_json_response(
                        category_response.content,
                        "category",
                        expected_keys=["name", "documentId"],
                    )
                    if parsed_category:
                        product["category"] = parsed_category
                    else:
                        print(
                            f"     ⚠️ Failed to parse category for product {index}/{total_products}"
                        )
                        # Try fallback: ask AI to choose from a simplified list
                        fallback_category = self._fallback_category_selection(
                            product, categories
                        )
                        if fallback_category:
                            product["category"] = fallback_category
                            print(
                                f"     ✅ Fallback category selected: {fallback_category}"
                            )
                        else:
                            print(
                                f"     ❌ No category could be determined for product {index}/{total_products}"
                            )
                            continue

                    sub_categories = self.get_sub_categories(product["category"])
                    sub_category_system_message = SystemMessage(
                        content=f"""You are a product sub categorization AI. You will receive a product JSON object containing name, description, materials, features, and other details, along with an array of objects with name and documentId of sub categories.

CRITICAL REQUIREMENTS:
- You MUST respond with ONLY valid JSON format
- Do NOT include any explanatory text, reasoning, or additional content
- Do NOT wrap your response in markdown code blocks
- Your response must be a single JSON object with "name" and "documentId" fields

TASK: Analyze the product's primary function, materials, and intended use to select the most appropriate sub category from the provided list. Prioritize the product's main purpose over secondary attributes. If multiple sub categories could apply, choose the most specific one.

RESPONSE FORMAT (return exactly this structure):
{{"name": "Selected Sub Category Name", "documentId": "selected_sub_category_document_id"}}

Available sub categories: {sub_categories}

Remember: Respond with ONLY the JSON object, nothing else."""
                    )

                    # Get AI response with retry logic for sub-category
                    sub_category_response = None

                    for attempt in range(max_retries):
                        try:
                            sub_category_response = self.ai_model.invoke(
                                [sub_category_system_message, human_message]
                            )
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(
                                    f"     ⚠️ Sub-category API call failed (attempt {attempt + 1}), retrying in 5 seconds..."
                                )
                                time.sleep(5)
                                continue
                            else:
                                print(
                                    f"     ❌ Sub-category API call failed after {max_retries} attempts: {e}"
                                )
                                raise e

                    if sub_category_response:
                        parsed_sub_category = parse_ai_json_response(
                            sub_category_response.content,
                            "sub-category",
                            expected_keys=["name", "documentId"],
                        )
                        if parsed_sub_category:
                            product["sub_category"] = parsed_sub_category
                            print(
                                f"     ✅ Added category: {product['category'].get('name')} | sub-category: {product['sub_category'].get('name')}"
                            )

                            # Update only this specific product in the JSON file
                            self._update_single_product_in_json(
                                json_path, index - 1, product
                            )
                        else:
                            print(
                                f"     ⚠️ Failed to parse sub-category for product {index}/{total_products}"
                            )
                            # Try fallback: ask AI to choose from a simplified list
                            fallback_sub_category = (
                                self._fallback_sub_category_selection(
                                    product, sub_categories
                                )
                            )
                            if fallback_sub_category:
                                product["sub_category"] = fallback_sub_category
                                print(
                                    f"     ✅ Fallback sub-category selected: {fallback_sub_category}"
                                )
                            else:
                                print(
                                    f"     ⚠️ No sub-category could be determined for product {index}/{total_products}"
                                )
                            # Still update with just the category
                            self._update_single_product_in_json(
                                json_path, index - 1, product
                            )

            except Exception as e:
                print(
                    f"     ❌ Error adding category to product {index}/{total_products}: {e}"
                )

    def _fallback_category_selection(self, product: dict, categories: list) -> dict:
        """Fallback method to select category when AI response parsing fails"""
        try:
            if not categories:
                return None

            # Create a simplified prompt with just the product name and a numbered list
            product_name = product.get("name", "Unknown Product")
            category_list = "\n".join(
                [
                    f"{i+1}. {cat.get('name', 'Unknown')}"
                    for i, cat in enumerate(categories)
                ]
            )

            fallback_message = SystemMessage(
                content=f"""You are a product categorization AI. The previous response failed to parse, so I need you to respond with ONLY a number.

Product: {product_name}

Available categories (choose by number):
{category_list}

Respond with ONLY the number (1, 2, 3, etc.) corresponding to the most appropriate category. Do not include any other text."""
            )

            human_message = HumanMessage(
                content="Please select the most appropriate category number."
            )

            response = self.ai_model.invoke([fallback_message, human_message])
            response_text = response.content.strip()

            # Extract number from response
            number_match = re.search(r"\b(\d+)\b", response_text)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(categories):
                    return categories[selected_index]

        except Exception as e:
            print(f"     ⚠️ Fallback category selection failed: {e}")

        return None

    def _fallback_sub_category_selection(
        self, product: dict, sub_categories: list
    ) -> dict:
        """Fallback method to select sub-category when AI response parsing fails"""
        try:
            if not sub_categories:
                return None

            # Create a simplified prompt with just the product name and a numbered list
            product_name = product.get("name", "Unknown Product")
            sub_category_list = "\n".join(
                [
                    f"{i+1}. {cat.get('name', 'Unknown')}"
                    for i, cat in enumerate(sub_categories)
                ]
            )

            fallback_message = SystemMessage(
                content=f"""You are a product sub-categorization AI. The previous response failed to parse, so I need you to respond with ONLY a number.

Product: {product_name}

Available sub-categories (choose by number):
{sub_category_list}

Respond with ONLY the number (1, 2, 3, etc.) corresponding to the most appropriate sub-category. Do not include any other text."""
            )

            human_message = HumanMessage(
                content="Please select the most appropriate sub-category number."
            )

            response = self.ai_model.invoke([fallback_message, human_message])
            response_text = response.content.strip()

            # Extract number from response
            number_match = re.search(r"\b(\d+)\b", response_text)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(sub_categories):
                    return sub_categories[selected_index]

        except Exception as e:
            print(f"     ⚠️ Fallback sub-category selection failed: {e}")

        return None

    def _update_single_product_in_json(
        self, json_path: str, product_index: int, updated_product: dict
    ):
        """Update a single product in the JSON file without rewriting the entire file"""
        try:
            # Read the current JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                products = json.load(f)

            # Update only the specific product
            products[product_index] = updated_product

            # Write back the updated products array
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(products, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"     ⚠️ Failed to update product {product_index + 1} in JSON: {e}")

    def get_categories(self):
        categories = []
        try:
            query = """
                query ProductCategories($pagination: PaginationArg) {
                    productCategories(pagination: $pagination) {
                        name
                        documentId
                    }
                }
            """

            # Variables to send with the query
            variables = {"pagination": {"limit": -1}}
            result = graphql_requester(query, variables)
            categories = result.get("productCategories")
        except Exception as e:
            print(str(e))
        return categories

    def get_sub_categories(self, category: dict):
        # Check cache first
        category_id = category.get("documentId")
        if category_id in self.sub_categories_cache:
            print(f"Using cached sub-categories for: {category.get('name')}")
            return self.sub_categories_cache[category_id]

        sub_categories = []

        try:
            print(f"Fetching sub-categories from API for: {category.get('name')}")
            query = """
                query ProductSubCategories($filters: ProductSubCategoryFiltersInput) {
                    productSubCategories(filters: $filters) {
                        name
                        documentId
                    }
                }
            """

            # Variables to send with the query
            variables = {
                "filters": {
                    "product_category": {
                        "documentId": {"eq": category.get("documentId")}
                    }
                }
            }
            result = graphql_requester(query, variables)
            sub_categories = result.get("productSubCategories")

            # Cache the result for future use
            self.sub_categories_cache[category_id] = sub_categories
            print(f"Cached sub-categories for: {category.get('name')}")

        except Exception as e:
            print(str(e))
        return sub_categories


# Usage Example
def main(
    pdf_file, use_ai: bool = True, add_category: bool = False, start_from: int = 0
):
    # Timing: record start time
    start_ts = time.time()
    start_dt = datetime.now()
    print(f"⏱️  Start time: {start_dt.isoformat()}")
    # Get Gemini API key (you can set this as environment variable or pass directly)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Respect environment variable to force-disable AI
    env_no_ai = os.getenv("PDF_EXTRACTOR_NO_AI", "").lower()
    if env_no_ai in ("1", "true", "yes", "y"):
        print("⚠️  PDF_EXTRACTOR_NO_AI is set — AI will be disabled.")
        use_ai = False

    if not gemini_api_key:
        print(
            "⚠️  GEMINI_API_KEY not found. Set it as environment variable or modify the code."
        )
        print("   Example: export GEMINI_API_KEY='your-api-key-here'")
        print("   AI features will be disabled.")
        # Disable AI if no API key
        use_ai = False

    extractor = PDFStructureExtractor(gemini_api_key=gemini_api_key)

    if add_category:
        print(f"Adding category to {pdf_file}")
        extractor.add_product_category(pdf_file, start_from)
    else:
        print(pdf_file)

        # Step 1: Create organized output folder structure
        print("Creating output folder structure...")
        paths = extractor.create_output_structure(pdf_file)

        print(f"Output directory created: {paths['main_dir']}")

        # Step 2: Convert PDF to structured XML
        print("Converting PDF to XML...")
        xml_file = extractor.pdf_to_xml(
            pdf_file, paths["xml_path"], paths["images_dir"]
        )
        print(f"XML saved to: {xml_file}")

        # Filter out small images before AI processing
        print("Filtering small images (<30px) from images folder and XML...")
        try:
            removed = extractor.prepare_for_ai(
                xml_path=xml_file,
                image_folder=paths["images_dir"],
                min_dim=30,
                remove_files=True,
            )
            print(f"Filtered {len(removed)} small images.")
            if removed:
                sample = ", ".join(sorted(list(removed))[:20])
                more = "..." if len(removed) > 20 else ""
                print(f"Removed images (sample): {sample}{more}")
        except Exception as e:
            print(f"⚠️ Failed to filter small images: {e}")

        # Generate product visualization with colored bounding boxes
        print("\n🎨 Generating product visualization...")
        try:
            from utils.visualize_products import draw_bounding_boxes

            color_mapping = draw_bounding_boxes(
                xml_path=xml_file,
                output_folder=paths["images_dir"],
                line_width=3,
                update_xml=True,
            )
            print(
                f"✅ Visualization complete! Generated {len(color_mapping) if color_mapping else 0} colored boxes"
            )
        except Exception as e:
            print(f"⚠️ Failed to generate visualization: {e}")
            import traceback

            traceback.print_exc()

        # Step 3: Extract products with AI enhancement (optional)
        if use_ai and extractor.ai_model:
            print("🤖 Starting AI-enhanced extraction...")
            ai_results = extractor.ai_enhanced_extraction(xml_file, paths)

            print(f"🤖 AI analysis complete!")
            print(
                f"   📊 AI Enhanced data: {paths['xml_path'].replace('_structure.xml', '_ai_enhanced.json')}"
            )
            print(
                f"   📈 Progress tracking: {paths['xml_path'].replace('_structure.xml', '_progress.json')}"
            )
        else:
            # AI disabled or AI model unavailable — produce JSON from XML for downstream tooling
            try:
                json_out = extractor.xml_to_json(xml_file)
                print(f"✅ Converted XML to JSON: {json_out}")
            except Exception as e:
                print(f"⚠️ Failed to convert XML to JSON: {e}")

            # If grouping helper is available, compute groups per page using proximity threshold
            if _group_blocks_module is not None and "json_out" in locals():
                try:
                    import json as _json

                    with open(json_out, "r", encoding="utf-8") as _f:
                        _j = _json.load(_f)

                    try:
                        thresh = float(os.getenv("PDF_GROUP_THRESHOLD", "5"))
                    except Exception:
                        thresh = 50.0

                    total_extra = 0
                    for _p in _j.get("pages", []):
                        try:
                            grp = _group_blocks_module.group_page(_p, threshold=thresh)
                        except Exception:
                            grp = []
                        # Run the post-processing attachments stage so the grouped
                        # JSON created by the pipeline includes the same safety
                        # attachments run locally (and considers blocked images).
                        try:
                            new_grp, extra = (
                                _group_blocks_module.attach_remaining_groups(
                                    grp, threshold=thresh, page=_p
                                )
                            )
                        except Exception:
                            new_grp = grp
                            extra = 0
                        _p["groups"] = new_grp
                        if extra:
                            total_extra += extra

                    grouped_out = (
                        os.path.splitext(json_out)[0] + ".fullblocks.grouped.json"
                    )
                    with open(grouped_out, "w", encoding="utf-8") as _f:
                        _json.dump(_j, _f, indent=2, ensure_ascii=False)
                    print(f"✅ Grouped JSON saved: {grouped_out}")
                    if total_extra:
                        print(
                            f"ℹ️  Post-processor made {total_extra} extra attachments across pages"
                        )
                except Exception as e:
                    print(f"⚠️ Failed to auto-group JSON: {e}")
            else:
                print("group_blocks not available — skipping grouping")

            print("⚠️  AI not available, only XML structure created.")

        print(f"\n✅ Processing complete!")
        print(f"📁 All files organized in: {paths['main_dir']}")
        print(f"   📄 XML: {os.path.basename(paths['xml_path'])}")

        if use_ai and extractor.ai_model:
            print(
                f"   🤖 AI Enhanced JSON: {os.path.basename(paths['xml_path'].replace('_structure.xml', '_ai_enhanced.json'))}"
            )
            print(
                f"   📈 Progress JSON: {os.path.basename(paths['xml_path'].replace('_structure.xml', '_progress.json'))}"
            )

        # Count images
        if os.path.exists(paths["images_dir"]):
            image_count = len(
                [f for f in os.listdir(paths["images_dir"]) if f.endswith(".png")]
            )
            print(f"   🖼️  Images folder: {image_count} images")

        print(f"\n🎉 Ready to explore your data!")
        if extractor.ai_model:
            print("   🤖 AI has intelligently matched images with related text")
            print("   📊 The XML contains all structural data with coordinates")
            print("   📈 Progress is saved after each page completion")

    # Timing: record end time and total elapsed
    end_ts = time.time()
    end_dt = datetime.now()
    total_seconds = end_ts - start_ts
    # Human friendly elapsed
    hrs, rem = divmod(int(total_seconds), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n⏱️  End time: {end_dt.isoformat()}")
    print(f"⏱️  Total elapsed time: {hrs}h {mins}m {secs}s ({total_seconds:.2f}s)")


if __name__ == "__main__":
    import sys

    # Defaults
    pdf_path = "./input/Villeroy.pdf"
    use_ai_flag = True
    add_category_flag = False
    start_from_flag = 0
    # Parse simple CLI: first positional arg is PDF path; use --no-ai to disable AI

    # add one condition for flag to directly start with adding category
    for a in sys.argv[1:]:
        if a in ("--no-ai", "--no_ai", "-n"):
            use_ai_flag = False
        elif a in ("--add-category", "--add_category", "-c"):
            add_category_flag = True
        elif (
            a.startswith("--start-from=")
            or a.startswith("--start_from=")
            or a.startswith("-s=")
        ):
            start_from_flag = int(a.split("=")[1])
        elif not a.startswith("-"):
            pdf_path = a

    print(f"Debug: start_from_flag = {start_from_flag}")
    main(
        pdf_path,
        use_ai=use_ai_flag,
        add_category=add_category_flag,
        start_from=start_from_flag,
    )
