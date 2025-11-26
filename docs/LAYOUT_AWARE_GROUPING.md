```markdown
# Layout-Aware Grouping System

## Overview

The grouping system **FIRST** detects the layout pattern of each PDF page, then applies a completely customized grouping algorithm optimized for that specific pattern. Layout detection happens before any grouping begins.

## Workflow

1. **Detect Layout** → Analyze spatial distribution of images and text
2. **Configure Algorithm** → Set pattern-specific parameters
3. **Apply Grouping** → Use optimized strategy for detected layout
4. **Post-process** → Safe attachment of remaining elements

## Detected Layout Types (11 Patterns)

### 1. **CATALOG_GRID Layout**

- **Pattern**: Wide product images in grid formation with captions below/beside
- **Detection**:
  - 4+ images spanning >60% page width
  - Average image width >25% of page
  - Similar image widths (variance <30%)
- **Strategy**:
  - Very aggressive horizontal grouping
  - radius_multiplier: 3.5, directional_multiplier: 6.0
  - MAX_TEXTS_PER_GROUP: 12, MAX_CHARS_PER_GROUP: 1500
- **Use case**: Product catalogs, retail flyers

### 2. **TABLE_GRID Layout**

- **Pattern**: Regular grid of similar-sized image/text cells
- **Detection**:
  - 6+ images with similar widths (variance <40%)
  - Forms distinct rows and columns (bucketed positions)
- **Strategy**:
  - Very conservative, keep cells separate
  - radius_multiplier: 1.2, directional_multiplier: 2.0
  - MAX_TEXTS_PER_GROUP: 4, MAX_CHARS_PER_GROUP: 400
- **Use case**: Data sheets, comparison tables, gallery layouts

### 3. **MULTI_COLUMN Layout**

- **Pattern**: Multiple text columns with interspersed images
- **Detection**:
  - Texts outnumber images 3:1
  - Text X-positions cluster into 2+ distinct columns (>15% each)
- **Strategy**:
  - Respect column boundaries, vertical grouping
  - radius_multiplier: 1.5, directional_multiplier: 3.0
  - MAX_TEXTS_PER_GROUP: 8, MAX_CHARS_PER_GROUP: 1000
- **Use case**: Newspapers, magazines, academic papers

### 4. **SIDEBAR Layout**

- **Pattern**: Images concentrated on one side, text on the other
- **Detection**: >75% of images in left or right third of page
- **Strategy**:
  - Wide horizontal search to bridge content divide
  - radius_multiplier: 5.0, directional_multiplier: 8.0
  - MAX_TEXTS_PER_GROUP: 10, MAX_CHARS_PER_GROUP: 1200
- **Use case**: Technical documentation, textbooks with margin figures

### 5. **CENTERED_CLUSTER Layout**

- **Pattern**: Images grouped in center area, text surrounds them
- **Detection**: >60% of images within center 60% of page (both X and Y)
- **Strategy**:
  - Conservative, small focused groups
  - radius_multiplier: 1.5, directional_multiplier: 2.0
  - MAX_TEXTS_PER_GROUP: 5, MAX_CHARS_PER_GROUP: 600
- **Use case**: Magazine articles, feature layouts, posters

### 6. **VERTICAL_FLOW Layout**

- **Pattern**: Content flows top-to-bottom, images stacked vertically
- **Detection**:
  - Vertical span >50% page height
  - Horizontal span <40% page width
- **Strategy**:
  - Prioritize vertical proximity over horizontal
  - radius_multiplier: 2.5, directional_multiplier: 4.0
  - MAX_TEXTS_PER_GROUP: 8, MAX_CHARS_PER_GROUP: 1000
- **Use case**: Mobile-optimized layouts, narrow columns, timelines

### 7. **HORIZONTAL_STRIPS Layout**

- **Pattern**: Content organized in horizontal bands across page
- **Detection**:
  - Horizontal span >70% page width
  - Vertical span <30% page height
- **Strategy**:
  - Prioritize horizontal proximity
  - radius_multiplier: 4.0, directional_multiplier: 7.0
  - MAX_TEXTS_PER_GROUP: 10, MAX_CHARS_PER_GROUP: 1200
- **Use case**: Infographics, timelines, process diagrams

### 8. **SCATTERED_DENSE Layout**

- **Pattern**: Many small elements packed densely
- **Detection**: Element density >5.0 per 10,000px²
- **Strategy**:
  - Tight grouping to separate closely-packed items
  - radius_multiplier: 1.3, directional_multiplier: 2.0
  - MAX_TEXTS_PER_GROUP: 6, MAX_CHARS_PER_GROUP: 700
- **Use case**: Dense dashboards, crowded layouts, icon sheets

### 9. **SCATTERED_SPARSE Layout**

- **Pattern**: Few elements spread across large area
- **Detection**: Element density <1.5 per 10,000px²
- **Strategy**:
  - Wide search radius to connect distant elements
  - radius_multiplier: 3.0, directional_multiplier: 5.0
  - MAX_TEXTS_PER_GROUP: 9, MAX_CHARS_PER_GROUP: 1100
- **Use case**: Minimal designs, spacious layouts, presentation slides

### 10. **FULL_PAGE_SINGLE Layout**

- **Pattern**: One or very few large dominant elements
- **Detection**:
  - ≤3 total elements
  - Average image width >50% page width
- **Strategy**:
  - Very aggressive grouping, combine everything
  - radius_multiplier: 8.0, directional_multiplier: 12.0
  - MAX_TEXTS_PER_GROUP: 20, MAX_CHARS_PER_GROUP: 3000
- **Use case**: Cover pages, full-page ads, single images with descriptions

### 11. **MIXED_DENSITY Layout**

- **Pattern**: Areas of dense and sparse content mixed together
- **Detection**:
  - High variance in inter-element distances
  - 4+ images with distance variance > average distance²
- **Strategy**:
  - Adaptive balanced approach
  - radius_multiplier: 2.5, directional_multiplier: 4.0
  - MAX_TEXTS_PER_GROUP: 8, MAX_CHARS_PER_GROUP: 900
  - **Use case**: Complex mixed layouts, combined content types

## Key Features

### 1. **No-Overlap Guarantee**

- Groups never have overlapping bounding boxes
- Checked at every attachment stage:
  - Initial text-to-image anchoring
  - Post-processing passes
  - Forced iterative matching
  - Final attachment pass

### 2. **Capacity Limits**

- Groups respect MAX_TEXTS_PER_GROUP and MAX_CHARS_PER_GROUP
- Prevents one group from swallowing too many text blocks
- Limits vary by layout type

### 3. **Iterative Refinement**

- Multiple passes ensure thorough grouping:
  1.  Initial DSU-based grouping (text-text, image-image, text-image)
  2.  Post-processing (text-only → image-only groups)
  3.  Forced iterative matching (cardinal directions only)
  4.  Final attachment pass
  5.  Safe remaining attachments (attach_remaining_groups)
- Repeats until no more attachments possible (max 12 iterations)

### 4. **Smart Selection**

- Prefers groups with fewer texts/chars when distances are similar
- One-text-per-target-per-iteration to prevent starvation
- Scoring: distance + TEXT_COUNT_WEIGHT × count + CHAR_COUNT_WEIGHT × chars

## Results on pdf_merged Dataset (50 pages)

### Layout Distribution
```

scattered_sparse : 11 pages ( 22.0%)
sidebar : 11 pages ( 22.0%)
scattered : 9 pages ( 18.0%)
centered_cluster : 9 pages ( 18.0%)
multi_column : 5 pages ( 10.0%)
table_grid : 4 pages ( 8.0%)
full_page_single : 1 page ( 2.0%)

````

### Grouping Quality

- **Average texts per group**: 2.6 (well-balanced)
- **Total groups with text**: 309
- **Large groups (>10 texts)**: Only 7 groups across 50 pages (2.3%)
- **Extra safe attachments**: 1 attachment on 1 page (minimal post-processing needed)

### Improvements vs Previous Version

- 6 pages improved (fewer orphaned text blocks)
- 4 pages slightly worse (layout detection being conservative)
- 40 pages unchanged
- Overall: Better detection granularity with 11 layout types vs 4

## Usage

### Basic Usage

```bash
python3 group_blocks.py input.json --out output.json
````

### With Custom Threshold

```bash
python3 group_blocks.py input.json --out output.json --threshold 10
```

### Check Layout Detection Results

```bash
python3 check_layouts.py output.json
```

## Technical Implementation

### Files Modified

- `group_blocks.py`: Main grouping logic with layout detection

### Key Functions

- `detect_page_layout(page)`: Analyzes page and returns layout type
- `group_page(page, threshold, mode)`: Groups blocks using layout-aware strategy
- `attach_remaining_groups(groups, threshold)`: Post-processor for safe attachments

### Debug Mode

Set `DEBUG_GROUPER = True` in `group_blocks.py` to see detailed skip reasons:

- Capacity overflow skips
- Bbox overlap skips
- Distance-based skips

## Output Format

Each page in the output JSON includes:

- `groups`: Array of grouped blocks
- `_detected_layout`: Layout type (for debugging)

Each group contains:

- `id`: Unique group identifier
- `bbox`: Union bounding box [x0, y0, x1, y1]
- `images`: Array of image objects
- `texts`: Array of text objects

## Future Improvements

1. **More Layout Types**: Add detection for:

   - Multi-column text with inline images
   - Table-based layouts
   - Full-bleed backgrounds with overlaid content

2. **Adaptive Thresholds**: Auto-tune thresholds based on:

   - Average element sizes
   - Page density
   - Element spacing patterns

3. **Machine Learning**: Train a classifier for layout detection using:

   - Spatial features
   - Element density maps
   - Aspect ratio distributions

4. **Confidence Scoring**: Add confidence scores to layout detection
   and fall back to conservative strategy when uncertain

```

```
