from utils.text_filter import DynamicTextFilter


def load_exact_data():
    page_data = {
        1: {5.0: 1, 12.0: 1, 10.0: 1, 7.0: 6, 2.7: 6},
        2: {5.0: 2, 10.0: 2, 7.0: 17, 3.3: 9, 4.0: 6, 12.0: 1},
        3: {5.0: 2, 12.0: 3, 7.0: 12, 3.2: 3, 10.0: 2, 3.3: 9, 4.0: 6},
    }
    return page_data


def convert_to_blocks_format(page_data):
    blocks = []
    block_id_counter = 0
    for page_num, font_counts in page_data.items():
        for font_size, count in font_counts.items():
            for i in range(count):
                text_length = 10 if font_size >= 8.0 else 5
                blocks.append(
                    {
                        "page_num": page_num,
                        "font_size": font_size,
                        "text_length": text_length,
                        "text": f"Sample text with font size {font_size} on page {page_num}",
                        "block_id": f"block_{block_id_counter}",
                    }
                )
                block_id_counter += 1
    return blocks


def test_dynamic_text_filter_basic():
    page_data = load_exact_data()
    blocks = convert_to_blocks_format(page_data)

    filter_system = DynamicTextFilter(confidence_threshold=0.6)
    filtered_blocks = filter_system.filter_text_blocks(blocks)

    assert isinstance(filtered_blocks, list)
    assert len(filtered_blocks) <= len(blocks)
