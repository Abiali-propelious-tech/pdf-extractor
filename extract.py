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
from requester import graphql_requester
from json_parser import parse_ai_json_response

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

            for i, b in enumerate(blocks):
                try:
                    x0, y0, x1, y1, text, *_ = b
                except Exception:
                    continue
                tb = ET.Element("text_block")
                tb.set("id", f"text_{page_num}_{i}")
                tb.set("bbox", f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}")
                # rudimentary font/size detection not available here; leave blank
                tb.set("font", "")
                tb.set("size", "")
                # classify type heuristically
                ttype = self._classify_text_type(text, 0, (x0, y0, x1, y1))
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
            
        except Exception:
            image_list = []

        # Track processed xrefs to avoid duplicates
        processed_xrefs = set()

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
                valid_rects = []
                for rect in img_rects:
                    # Check if rectangle is within page boundaries with some tolerance
                    tolerance = 1.0  # PDF points
                    if (rect.x0 >= page_rect.x0 - tolerance and 
                        rect.y0 >= page_rect.y0 - tolerance and 
                        rect.x1 <= page_rect.x1 + tolerance and 
                        rect.y1 <= page_rect.y1 + tolerance):
                        # Also check if rectangle has reasonable dimensions
                        if rect.width > 1 and rect.height > 1:
                            valid_rects.append(rect)
                
                # Skip if no valid rectangles found on this page
                if not valid_rects:
                    continue
                    
                # Mark this xref as processed
                processed_xrefs.add(xref)
                
                # Process each valid rectangle occurrence
                for r_idx, rect in enumerate(valid_rects):
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
                            min(rect.y1 + pad, page_rect.y1)
                        )

                        # Render at higher resolution for better quality
                        zoom = 2  # 2x resolution
                        mat = fitz.Matrix(zoom, zoom)
                        
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

                        # Create XML element only if extraction was successful
                        if extraction_success:
                            img_block = ET.Element("image_block")
                            img_block.set("id", f"img_{page_num}_{img_index}_{r_idx}")
                            img_block.set(
                                "bbox",
                                f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}",
                            )
                            img_block.set("width", f"{rect.width:.1f}")
                            img_block.set("height", f"{rect.height:.1f}")
                            img_block.set("file", image_filename)
                            img_block.set("xref", str(xref))  # Add xref for debugging
                            page_elem.append(img_block)
                        else:
                            # Create element marking extraction failure
                            img_block = ET.Element("image_block")
                            img_block.set("id", f"img_{page_num}_{img_index}_{r_idx}")
                            img_block.set(
                                "bbox",
                                f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}",
                            )
                            img_block.set("width", f"{rect.width:.1f}")
                            img_block.set("height", f"{rect.height:.1f}")
                            img_block.set("file", "extraction_failed")
                            img_block.set("xref", str(xref))
                            page_elem.append(img_block)

                    except Exception as e:
                        print(f"⚠️ Failed to process image {img_index}_{r_idx} on page {page_num+1}: {e}")
                        continue

            except Exception as e:
                print(f"⚠️ Failed to process image {img_index} on page {page_num+1}: {e}")
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
                f"  - Font: {text.get('font', '')} (Size: {text.get('size', '')})\n\n"
            )

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
            formatted += f"  - Position: {text['bbox']}\n\n"

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
            variables = {
                "pagination": {
                    "limit": -1  
                }
            }

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

    def filter_small_images(
        self,
        image_folder: str,
        xml_path: str,
        min_dim: int = 30,
        remove_files: bool = True,
    ) -> set:
        """
        Scan `image_folder` for common image files, delete those where width < min_dim or height < min_dim,
        and remove references to those files from the XML at `xml_path`.
        Returns a set of basenames that were removed (or would be removed).
        """
        removed_basenames = set()

        if not os.path.isdir(image_folder):
            return removed_basenames

        patterns = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.gif", "*.webp")

        for pat in patterns:
            for path in glob.glob(os.path.join(image_folder, pat)):
                try:
                    # Prefer PIL for accurate size
                    from PIL import Image as _PILImage

                    with _PILImage.open(path) as _im:
                        w, h = _im.size
                except Exception:
                    # If PIL not available or fails, try to approximate via file size or skip
                    try:
                        import fitz

                        pix = fitz.Pixmap(path)
                        w, h = getattr(pix, "width", 0), getattr(pix, "height", 0)
                    except Exception:
                        # Can't determine size -> skip conservative
                        continue

                if w < min_dim or h < min_dim:
                    removed_basenames.add(os.path.basename(path))
                    if remove_files:
                        try:
                            os.remove(path)
                        except Exception:
                            pass

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

    def add_product_category(self, json_path: str, start_from = 0):
        """Add product category to the JSON file"""
        categories = self.get_categories()

        with open(json_path, "r", encoding="utf-8") as f:
            products = json.load(f)

        total_products = len(products)
        print(f"Debug: start_from parameter = {start_from}")
        print(f"🏷️  Starting categorization for {total_products} products...")

        system_message = SystemMessage(content=f"""You are a product categorization AI. You will receive a product JSON object containing name, description, materials, features, and other details, along with an array of objects with name and documentId of categories. 

CRITICAL REQUIREMENTS:
- You MUST respond with ONLY valid JSON format
- Do NOT include any explanatory text, reasoning, or additional content
- Do NOT wrap your response in markdown code blocks
- Your response must be a single JSON object with "name" and "documentId" fields

TASK: Analyze the product's primary function, materials, and intended use to select the most appropriate category from the provided list. Prioritize the product's main purpose over secondary attributes. If multiple categories could apply, choose the most specific one.

RESPONSE FORMAT (return exactly this structure):
{{"name": "Selected Category Name", "documentId": "selected_category_document_id"}}

Available categories: {categories}

Remember: Respond with ONLY the JSON object, nothing else.""")

        for index, product in enumerate(products[start_from:], start_from + 1):
            try:
                print(f"   📝 Processing product {index}/{total_products}: {product.get('name', 'Unknown')}")
                
                human_message = HumanMessage(content=f"""Product: {product}""")
                
                # Get AI response with retry logic for category
                import time
                max_retries = 3
                category_response = None
                
                for attempt in range(max_retries):
                    try:
                        category_response = self.ai_model.invoke([system_message, human_message])
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"     ⚠️ Category API call failed (attempt {attempt + 1}), retrying in 5 seconds...")
                            time.sleep(5)
                            continue
                        else:
                            print(f"     ❌ Category API call failed after {max_retries} attempts: {e}")
                            raise e
                
                if category_response:
                    parsed_category = parse_ai_json_response(category_response.content, "category", expected_keys=["name", "documentId"])
                    if parsed_category:
                        product["category"] = parsed_category
                    else:
                        print(f"     ⚠️ Failed to parse category for product {index}/{total_products}")
                        # Try fallback: ask AI to choose from a simplified list
                        fallback_category = self._fallback_category_selection(product, categories)
                        if fallback_category:
                            product["category"] = fallback_category
                            print(f"     ✅ Fallback category selected: {fallback_category}")
                        else:
                            print(f"     ❌ No category could be determined for product {index}/{total_products}")
                            continue

                    sub_categories = self.get_sub_categories(product["category"])
                    sub_category_system_message = SystemMessage(content=f"""You are a product sub categorization AI. You will receive a product JSON object containing name, description, materials, features, and other details, along with an array of objects with name and documentId of sub categories.

CRITICAL REQUIREMENTS:
- You MUST respond with ONLY valid JSON format
- Do NOT include any explanatory text, reasoning, or additional content
- Do NOT wrap your response in markdown code blocks
- Your response must be a single JSON object with "name" and "documentId" fields

TASK: Analyze the product's primary function, materials, and intended use to select the most appropriate sub category from the provided list. Prioritize the product's main purpose over secondary attributes. If multiple sub categories could apply, choose the most specific one.

RESPONSE FORMAT (return exactly this structure):
{{"name": "Selected Sub Category Name", "documentId": "selected_sub_category_document_id"}}

Available sub categories: {sub_categories}

Remember: Respond with ONLY the JSON object, nothing else.""")
                    
                    # Get AI response with retry logic for sub-category
                    sub_category_response = None
                    
                    for attempt in range(max_retries):
                        try:
                            sub_category_response = self.ai_model.invoke([sub_category_system_message, human_message])
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"     ⚠️ Sub-category API call failed (attempt {attempt + 1}), retrying in 5 seconds...")
                                time.sleep(5)
                                continue
                            else:
                                print(f"     ❌ Sub-category API call failed after {max_retries} attempts: {e}")
                                raise e
                    
                    if sub_category_response:
                        parsed_sub_category = parse_ai_json_response(sub_category_response.content, "sub-category", expected_keys=["name", "documentId"])
                        if parsed_sub_category:
                            product["sub_category"] = parsed_sub_category
                            print(f"     ✅ Added category: {product['category'].get('name')} | sub-category: {product['sub_category'].get('name')}")
                            
                            # Update only this specific product in the JSON file
                            self._update_single_product_in_json(json_path, index - 1, product)
                        else:
                            print(f"     ⚠️ Failed to parse sub-category for product {index}/{total_products}")
                            # Try fallback: ask AI to choose from a simplified list
                            fallback_sub_category = self._fallback_sub_category_selection(product, sub_categories)
                            if fallback_sub_category:
                                product["sub_category"] = fallback_sub_category
                                print(f"     ✅ Fallback sub-category selected: {fallback_sub_category}")
                            else:
                                print(f"     ⚠️ No sub-category could be determined for product {index}/{total_products}")
                            # Still update with just the category
                            self._update_single_product_in_json(json_path, index - 1, product)
                        
            except Exception as e:
                print(f"     ❌ Error adding category to product {index}/{total_products}: {e}")

    def _fallback_category_selection(self, product: dict, categories: list) -> dict:
        """Fallback method to select category when AI response parsing fails"""
        try:
            if not categories:
                return None
                
            # Create a simplified prompt with just the product name and a numbered list
            product_name = product.get('name', 'Unknown Product')
            category_list = "\n".join([f"{i+1}. {cat.get('name', 'Unknown')}" for i, cat in enumerate(categories)])
            
            fallback_message = SystemMessage(content=f"""You are a product categorization AI. The previous response failed to parse, so I need you to respond with ONLY a number.

Product: {product_name}

Available categories (choose by number):
{category_list}

Respond with ONLY the number (1, 2, 3, etc.) corresponding to the most appropriate category. Do not include any other text.""")
            
            human_message = HumanMessage(content="Please select the most appropriate category number.")
            
            response = self.ai_model.invoke([fallback_message, human_message])
            response_text = response.content.strip()
            
            # Extract number from response
            number_match = re.search(r'\b(\d+)\b', response_text)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(categories):
                    return categories[selected_index]
                    
        except Exception as e:
            print(f"     ⚠️ Fallback category selection failed: {e}")
        
        return None

    def _fallback_sub_category_selection(self, product: dict, sub_categories: list) -> dict:
        """Fallback method to select sub-category when AI response parsing fails"""
        try:
            if not sub_categories:
                return None
                
            # Create a simplified prompt with just the product name and a numbered list
            product_name = product.get('name', 'Unknown Product')
            sub_category_list = "\n".join([f"{i+1}. {cat.get('name', 'Unknown')}" for i, cat in enumerate(sub_categories)])
            
            fallback_message = SystemMessage(content=f"""You are a product sub-categorization AI. The previous response failed to parse, so I need you to respond with ONLY a number.

Product: {product_name}

Available sub-categories (choose by number):
{sub_category_list}

Respond with ONLY the number (1, 2, 3, etc.) corresponding to the most appropriate sub-category. Do not include any other text.""")
            
            human_message = HumanMessage(content="Please select the most appropriate sub-category number.")
            
            response = self.ai_model.invoke([fallback_message, human_message])
            response_text = response.content.strip()
            
            # Extract number from response
            number_match = re.search(r'\b(\d+)\b', response_text)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(sub_categories):
                    return sub_categories[selected_index]
                    
        except Exception as e:
            print(f"     ⚠️ Fallback sub-category selection failed: {e}")
        
        return None

    def _update_single_product_in_json(self, json_path: str, product_index: int, updated_product: dict):
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
            variables = {
                "pagination": {
                    "limit": -1  
                }
            }
            result = graphql_requester(query, variables)
            categories = result.get("productCategories")
        except Exception as e:
            print(str(e))
        return categories
    
    def get_sub_categories(self, category: dict):
        # Check cache first
        category_id = category.get('documentId')
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
                        "documentId": {
                            "eq": category.get('documentId')
                        }
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
def main(pdf_file, use_ai: bool = True, add_category: bool = False, start_from: int = 0):
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
        xml_file = extractor.pdf_to_xml(pdf_file, paths["xml_path"], paths["images_dir"])
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

    #add one condition for flag to directly start with adding category
    for a in sys.argv[1:]:
        if a in ("--no-ai", "--no_ai", "-n"):
            use_ai_flag = False
        elif a in ("--add-category", "--add_category", "-c"):
            add_category_flag = True
        elif a.startswith("--start-from=") or a.startswith("--start_from=") or a.startswith("-s="):
            start_from_flag = int(a.split("=")[1])
        elif not a.startswith("-"):
            pdf_path = a

    print(f"Debug: start_from_flag = {start_from_flag}")
    main(pdf_path, use_ai=use_ai_flag, add_category=add_category_flag, start_from=start_from_flag)



