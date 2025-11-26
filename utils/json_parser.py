"""
Reusable AI JSON Response Parser (moved to utils/)

This module provides a robust JSON parser specifically designed to handle AI responses
that may contain explanatory text, markdown code blocks, or malformed JSON.

Usage:
    from utils.json_parser import parse_ai_json_response

    result = parse_ai_json_response(ai_response, "product analysis")
    if result:
        print("Successfully parsed JSON:", result)
    else:
        print("Failed to parse JSON")
"""

import json
import re
from typing import Any, Optional, Dict, List, Union


def parse_ai_json_response(
    response_content: str,
    context_name: str = "AI response",
    expected_keys: Optional[List[str]] = None,
    verbose: bool = False,
) -> Optional[Union[Dict, List]]:
    """
    Parse AI JSON response with comprehensive error handling and sanitization.
    Handles markdown code blocks, trailing commas, control characters, and other common issues.
    """
    try:
        raw = (response_content or "").strip()

        if verbose:
            print(f"[{context_name}] Raw response: {raw[:200]}...")

        # If wrapped in triple-backticks, extract inner content
        if raw.startswith("```"):
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
            else:
                raw = re.sub(
                    r"^```(?:json)?\s*\n?|```\s*$", "", raw, flags=re.MULTILINE
                ).strip()

        # If the response contains explanatory text before JSON, extract just the JSON part
        # Look for JSON object pattern: { "key": "value" }
        if expected_keys:
            key_pattern = "|".join(re.escape(key) for key in expected_keys)
            json_match = re.search(rf'\{{[^\{{}}]*"(?:{key_pattern})"[^\{{}}]*\}}', raw)
            if json_match:
                raw = json_match.group(0).strip()
            else:
                json_array_match = re.search(
                    rf'\[[^\[\]]*\{{[^\{{}}]*"(?:{key_pattern})"[^\{{}}]*\}}[^\[\]]*\]',
                    raw,
                )
                if json_array_match:
                    raw = json_array_match.group(0).strip()
        else:
            json_array_match = re.search(r"\[[\s\S]*\]", raw)
            if json_array_match:
                raw = json_array_match.group(0).strip()
            else:
                json_match = re.search(r"\{[^{}]*\}", raw)
                if json_match:
                    raw = json_match.group(0).strip()

        if not raw or raw.strip() in ('""', "''"):
            if verbose:
                print(f"[{context_name}] Empty response")
            return None

        def _sanitize_json_text(s: str) -> str:
            first = s.find("{")
            first_arr = s.find("[")
            if first == -1 or (first_arr != -1 and first_arr < first):
                first = first_arr
            if first > 0:
                s = s[first:]

            last_obj = s.rfind("}")
            last_arr = s.rfind("]")
            last = max(last_obj, last_arr)
            if last != -1 and last < len(s) - 1:
                s = s[: last + 1]

            s = re.sub(r",\s*(?=[}\]])", "", s)
            s = re.sub(r",\s*,+", ",", s)

            return s.strip()

        def _escape_control_chars_in_json_strings(s: str) -> str:
            def _replace(match):
                txt = match.group(0)
                inner = txt[1:-1]
                new_inner = []
                i = 0
                while i < len(inner):
                    ch = inner[i]
                    if ch == "\\" and i + 1 < len(inner):
                        new_inner.append(ch)
                        i += 1
                        new_inner.append(inner[i])
                    else:
                        cp = ord(ch)
                        if cp < 0x20:
                            new_inner.append("\\u%04x" % cp)
                        else:
                            new_inner.append(ch)
                    i += 1
                return '"' + "".join(new_inner) + '"'

            return re.sub(r'"(\\.|[^"\\])*"', _replace, s)

        parsed = None

        try:
            parsed = json.loads(raw)
            if verbose:
                print(f"[{context_name}] Direct JSON parse successful")
        except Exception:
            candidate = None
            arr_match = re.search(r"(\[\s*[\s\S]*\])", raw)
            obj_match = re.search(r"(\{[\s\S]*\})", raw)
            if arr_match:
                candidate = arr_match.group(1)
            elif obj_match:
                candidate = obj_match.group(1)
            else:
                candidate = raw

            candidate = _sanitize_json_text(candidate)
            candidate = _escape_control_chars_in_json_strings(candidate)

            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    cand2 = candidate.replace("'", '"')
                    cand2 = _sanitize_json_text(cand2)
                    cand2 = _escape_control_chars_in_json_strings(cand2)
                    parsed = json.loads(cand2)
                except Exception:
                    parsed = None

        if parsed is None:
            return None

        if expected_keys and isinstance(parsed, dict):
            missing_keys = [key for key in expected_keys if key not in parsed]
            if missing_keys and verbose:
                print(
                    f"[{context_name}] Warning: Missing expected keys: {missing_keys}"
                )

        return parsed

    except Exception:
        return None


def parse_ai_json_response_simple(response_content: str) -> Optional[Union[Dict, List]]:
    try:
        raw = response_content.strip()
        if raw.startswith("```"):
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()

        return json.loads(raw)
    except Exception:
        return None


def validate_json_structure(
    data: Union[Dict, List], required_keys: List[str] = None
) -> bool:
    if not data:
        return False

    if isinstance(data, dict) and required_keys:
        return all(key in data for key in required_keys)

    return True
