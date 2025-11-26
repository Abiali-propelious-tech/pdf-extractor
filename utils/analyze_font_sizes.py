import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import statistics


def analyze(xml_path="output/pdf_merged/pdf_merged_structure.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_font_sizes = defaultdict(list)

    for page in root.findall("page"):
        page_num = int(page.get("number"))
        for tb in page.findall("text_block"):
            size = tb.get("size")
            if size:
                try:
                    page_font_sizes[page_num].append(float(size))
                except ValueError:
                    pass

    unique_font_counts = []
    all_font_sizes = []
    font_size_counter = Counter()

    for page_num, sizes in page_font_sizes.items():
        unique_sizes = set(sizes)
        unique_font_counts.append(len(unique_sizes))
        all_font_sizes.extend(sizes)
        font_size_counter.update(unique_sizes)

    print("Font size distribution per page:")
    for page_num, sizes in sorted(page_font_sizes.items()):
        print(f"Page {page_num}: {sorted(set(sizes))}")

    print("\nSummary:")
    print(f"Total pages analyzed: {len(page_font_sizes)}")
    print(
        f"Min unique font sizes on a page: {min(unique_font_counts) if unique_font_counts else 0}"
    )
    print(
        f"Max unique font sizes on a page: {max(unique_font_counts) if unique_font_counts else 0}"
    )
    print(
        f"Average unique font sizes per page: {statistics.mean(unique_font_counts) if unique_font_counts else 0:.2f}"
    )
    print(f"Most common font sizes (overall): {font_size_counter.most_common(10)}")
    print(f"All font sizes (overall, sorted): {sorted(set(all_font_sizes))}")


if __name__ == "__main__":
    analyze()
