import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


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

    print("Per-page font size stats:\n")
    for page_num in sorted(page_font_sizes):
        sizes = page_font_sizes[page_num]
        size_counter = Counter(sizes)
        if size_counter:
            most_common_size, most_common_count = size_counter.most_common(1)[0]
            print(f"Page {page_num}: Font sizes count: {dict(size_counter)}")
            print(
                f"  Most common font size: {most_common_size} (count: {most_common_count})"
            )
            print(f"  Total text blocks: {len(sizes)}\n")
        else:
            print(f"Page {page_num}: No font size data found.\n")


if __name__ == "__main__":
    analyze()
