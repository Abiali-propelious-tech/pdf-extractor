import math
from typing import Dict, List, Tuple, Any
from collections import Counter


class DynamicTextFilter:
    """
    A dynamic, context-aware text filtering system for PDF content that adapts to each document's
    unique font distribution rather than using static font size ranges.
    """

    def __init__(
        self,
        font_deviation_weight=0.4,
        text_quality_weight=0.25,
        noise_weight=0.2,
        position_weight=0.15,
        confidence_threshold=0.6,
    ):
        self.font_deviation_weight = font_deviation_weight
        self.text_quality_weight = text_quality_weight
        self.noise_weight = noise_weight
        self.position_weight = position_weight
        self.confidence_threshold = confidence_threshold
        self.font_profile = {}

    def analyze_font_profile(
        self, page_font_data: Dict[int, Dict[float, int]]
    ) -> Dict[str, Any]:
        all_fonts = []
        for page_fonts in page_font_data.values():
            for font_size, count in page_fonts.items():
                all_fonts.extend([font_size] * count)

        if not all_fonts:
            return {}

        font_counts = Counter(all_fonts)
        sorted_fonts_by_count = font_counts.most_common()

        mean_font = sum(all_fonts) / len(all_fonts)
        sorted_all = sorted(all_fonts)
        median_font = sorted_all[len(sorted_all) // 2]

        variance = sum((x - mean_font) ** 2 for x in all_fonts) / len(all_fonts)
        std_dev = math.sqrt(variance)

        dominant_font, dominant_count = (
            sorted_fonts_by_count[0] if sorted_fonts_by_count else (0, 0)
        )

        content_range = [dominant_font - 2 * std_dev, dominant_font + 2 * std_dev]

        profile = {
            "mean_font_size": mean_font,
            "median_font_size": median_font,
            "std_dev": std_dev,
            "dominant_font": dominant_font,
            "dominant_count": dominant_count,
            "total_blocks": len(all_fonts),
            "content_font_range": content_range,
            "font_counts": font_counts,
            "total_pages": len(page_font_data),
        }

        self.font_profile = profile
        return profile

    def calculate_page_metrics(self, page_fonts: Dict[float, int]) -> Dict[str, Any]:
        if not page_fonts:
            return {"block_density": 0, "font_homogeneity": 0, "is_content_page": False}

        total_blocks = sum(page_fonts.values())
        font_counts = list(page_fonts.values())
        most_common_count = max(font_counts) if font_counts else 0

        page_density = total_blocks
        font_homogeneity = most_common_count / total_blocks if total_blocks > 0 else 0

        is_content_page = total_blocks > 5 and font_homogeneity > 0.3

        return {
            "block_density": page_density,
            "font_homogeneity": font_homogeneity,
            "is_content_page": is_content_page,
            "total_blocks": total_blocks,
        }

    def score_text_block(
        self, font_size: float, text_length: int, page_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.font_profile:
            raise ValueError("Must analyze font profile first")

        font_deviation = abs(font_size - self.font_profile["dominant_font"])
        if self.font_profile["std_dev"] > 0:
            normalized_deviation = font_deviation / self.font_profile["std_dev"]
        else:
            normalized_deviation = 0

        font_score = max(0, 1 - normalized_deviation / 3)

        if 5 <= text_length <= 100:
            text_quality_score = 1.0
        elif text_length < 5:
            text_quality_score = 0.3
        else:
            text_quality_score = 0.7

        noise_score = 1.0
        position_score = 1.0

        confidence = (
            font_score * self.font_deviation_weight
            + text_quality_score * self.text_quality_weight
            + noise_score * self.noise_weight
            + position_score * self.position_weight
        )

        return {
            "font_score": font_score,
            "text_quality_score": text_quality_score,
            "noise_score": noise_score,
            "position_score": position_score,
            "confidence": confidence,
            "is_content": confidence > self.confidence_threshold,
        }

    def filter_text_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not blocks:
            return []

        page_font_data = {}
        for block in blocks:
            page_num = block.get("page_num", 0)
            font_size = block.get("font_size", 0)

            if page_num not in page_font_data:
                page_font_data[page_num] = {}

            if font_size not in page_font_data[page_num]:
                page_font_data[page_num][font_size] = 0

            page_font_data[page_num][font_size] += 1

        self.analyze_font_profile(page_font_data)

        filtered_blocks = []

        for block in blocks:
            page_num = block.get("page_num", 0)
            page_fonts = page_font_data.get(page_num, {})
            page_metrics = self.calculate_page_metrics(page_fonts)
            scoring_result = self.score_text_block(
                block.get("font_size", 0), block.get("text_length", 0), page_metrics
            )

            block_with_score = block.copy()
            block_with_score.update(scoring_result)

            if scoring_result["is_content"]:
                filtered_blocks.append(block_with_score)

        return filtered_blocks


def test_with_sample_data():
    # Provide a minimal smoke-test harness (kept from original file)
    sample_page_data = {
        1: {5.0: 1, 12.0: 1, 10.0: 1, 7.0: 6, 2.7: 6},
        2: {5.0: 2, 10.0: 2, 7.0: 17, 3.3: 9, 4.0: 6, 12.0: 1},
        6: {12.0: 1, 20.5: 2, 8.0: 36, 10.0: 4, 21.2: 2, 5.0: 1},
        27: {10.0: 18, 1.3: 30, 18.0: 1, 8.0: 7, 9.0: 4, 4.0: 12, 4.5: 1},
        40: {7.0: 32},
    }

    blocks = []
    for page_num, font_counts in sample_page_data.items():
        for font_size, count in font_counts.items():
            for i in range(count):
                blocks.append(
                    {
                        "page_num": page_num,
                        "font_size": font_size,
                        "text_length": 10 if font_size >= 7.0 else 5,
                        "text": f"Sample text with font {font_size}",
                        "block_id": f"{page_num}_{font_size}_{i}",
                    }
                )

    filter_system = DynamicTextFilter(confidence_threshold=0.6)
    filtered_blocks = filter_system.filter_text_blocks(blocks)

    print(f"Total blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered_blocks)}")

    return filtered_blocks


if __name__ == "__main__":
    test_with_sample_data()
