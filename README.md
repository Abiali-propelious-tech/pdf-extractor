# PDF Product Extractor

AI-powered PDF extraction tool that identifies products from catalogue PDFs and visualizes them with bounding boxes.

## Features

- **PDF to XML Extraction**: Extracts structured data from PDFs including text blocks and images with coordinates
- **AI-Enhanced Analysis**: Uses Google Gemini AI to intelligently match products with their images and descriptions
- **Product Visualization**: Creates annotated images with colored bounding boxes around each product
- **Color Mapping**: Generates interactive HTML legends mapping box colors to product images
- **Auto-Categorization**: Automatically assigns categories and sub-categories to products

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY='your-api-key-here'
```

## Usage

### 1. Extract Products from PDF

```bash
# Basic extraction (with AI)
python extract.py path/to/your.pdf

# Without AI (XML structure only)
python extract.py path/to/your.pdf --no-ai

# Add categories to existing products
python extract.py path/to/products.json --add-category
```

### 2. Visualize Products with Bounding Boxes

```bash
# Visualize all PDFs in output folder
python visualize_products.py

# Visualize specific PDF folder
python visualize_products.py ./output/pdf_name
```

## Output Structure

```
output/
├── pdf_name/
│   ├── pdf_name_structure.xml          # Structured XML with all elements
│   ├── pdf_name_ai_products.json       # AI-extracted products
│   ├── pdf_name_progress.json          # Extraction progress
│   └── images/
│       ├── page_1_full.png             # Full page snapshot
│       ├── page_1_image_*.png          # Individual product images
│       ├── page_1_annotated.png        # ⭐ Full page with colored boxes
│       ├── page_1_color_mapping.json   # ⭐ Color to image mapping
│       └── page_1_legend.html          # ⭐ Interactive HTML legend
```

## Product Visualization Features

The visualization tool creates:

1. **Annotated Full Page Image** (`*_annotated.png`)

   - Full PDF page with colored bounding boxes around each product
   - Each box has a numbered label for easy identification
   - High-quality rendering at 2x resolution

2. **Color Mapping JSON** (`*_color_mapping.json`)

   ```json
   [
     {
       "index": 1,
       "color": "#e52d2d",
       "rgb": [229, 45, 45],
       "image_id": "img_0_0_0",
       "image_file": "page_1_image_0_0.png",
       "bbox": [813.7, 289.2, 945.6, 365.9],
       "width": "131.9",
       "height": "76.7"
     }
   ]
   ```

3. **Interactive HTML Legend** (`*_legend.html`)
   - Visual legend showing each product with its color
   - Thumbnail previews of each product image
   - Detailed information (ID, file, bbox, size)
   - Responsive grid layout

## Environment Variables

- `GEMINI_API_KEY` - Required for AI features
- `PDF_EXTRACTOR_NO_AI` - Set to `1` to disable AI
- `PDF_EXTRACTOR_DEBUG_IMAGES` - Set to `1` for detailed image extraction logs
- `PDF_EXTRACTOR_GHOST_FILTER` - Set to `1` to enable ghost/text image filtering
- `PDF_EXTRACTOR_GHOST_MODE` - `light`, `balanced`, or `strict` filtering
- `PDF_EXTRACTOR_MIN_RECT_SIDE_PTS` - Minimum image side in points (default: 25)
- `PDF_EXTRACTOR_MIN_RECT_AREA_FRAC` - Minimum area fraction (default: 0.001)

## How It Works

### Extraction Pipeline

1. **PDF Processing**: PyMuPDF extracts page content with coordinates
2. **Image Extraction**: Intelligent filtering removes duplicates and artifacts
3. **AI Analysis**: Gemini AI analyzes full page images to identify products
4. **Product Matching**: AI matches extracted images to product descriptions
5. **Categorization**: Products are automatically categorized

### Visualization Pipeline

1. **XML Parsing**: Reads bounding box coordinates from structure XML
2. **Color Generation**: Creates visually distinct colors for each product
3. **Box Drawing**: Draws colored rectangles on full page image
4. **Legend Creation**: Generates interactive HTML with color mappings
5. **Mapping Export**: Saves JSON data for programmatic access

## Example Workflow

```bash
# 1. Extract products from PDF
python extract.py ./input/catalogue.pdf

# 2. Visualize products with colored boxes
python visualize_products.py

# 3. Open the HTML legend in your browser
open ./output/catalogue/images/page_1_legend.html
```

## Color Mapping Use Cases

- **Quality Control**: Verify all products are detected correctly
- **Layout Analysis**: Understand product placement on pages
- **Training Data**: Create labeled datasets for ML models
- **Documentation**: Visual reference for product locations
- **Debugging**: Troubleshoot extraction issues

## Advanced Usage

### Custom Visualization

```python
from visualize_products import draw_bounding_boxes

# Generate visualization with custom line width
draw_bounding_boxes(
    xml_path='./output/pdf/pdf_structure.xml',
    output_folder='./output/pdf/images',
    line_width=5  # Thicker boxes
)
```

### Access Color Mapping Programmatically

```python
import json

# Load color mapping
with open('./output/pdf/images/page_1_color_mapping.json', 'r') as f:
    mapping = json.load(f)

# Find product by index
product = next(item for item in mapping if item['index'] == 1)
print(f"Product 1: {product['image_file']}")
print(f"Color: {product['color']}")
print(f"Bounding Box: {product['bbox']}")
```

## Requirements

- Python 3.8+
- PyMuPDF (fitz)
- Pillow (PIL)
- LangChain + Google Generative AI (for AI features)
- python-dotenv

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
