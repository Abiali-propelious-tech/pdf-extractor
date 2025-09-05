"""
Reusable AI JSON Response Parser

This module provides a robust JSON parser specifically designed to handle AI responses
that may contain explanatory text, markdown code blocks, or malformed JSON.

Usage:
    from json_parser import parse_ai_json_response
    
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
    verbose: bool = False
) -> Optional[Union[Dict, List]]:
    """
    Parse AI JSON response with comprehensive error handling and sanitization.
    Handles markdown code blocks, trailing commas, control characters, and other common issues.
    
    Args:
        response_content: The raw response content from AI
        context_name: Name for error messages (e.g., "product analysis", "category", "sub-category")
        expected_keys: Optional list of keys that should be present in the JSON object
        verbose: If True, prints detailed debugging information
        
    Returns:
        Parsed JSON object (dict or list) or None if parsing fails
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
            # Create a more specific pattern based on expected keys
            key_pattern = "|".join(re.escape(key) for key in expected_keys)
            json_match = re.search(rf'\{{[^{{}}]*"(?:{key_pattern})"[^{{}}]*\}}', raw)
            if json_match:
                raw = json_match.group(0).strip()
            else:
                # Also try to find JSON array pattern: [ { "key": "value" } ]
                json_array_match = re.search(rf'\[[^\[\]]*\{{[^{{}}]*"(?:{key_pattern})"[^{{}}]*\}}[^\[\]]*\]', raw)
                if json_array_match:
                    raw = json_array_match.group(0).strip()
        else:
            # Generic JSON object/array detection - try array first for product analysis
            json_array_match = re.search(r'\[[\s\S]*\]', raw)
            if json_array_match:
                raw = json_array_match.group(0).strip()
            else:
                json_match = re.search(r'\{[^{}]*\}', raw)
                if json_match:
                    raw = json_match.group(0).strip()

        # If empty or only quotes, bail out gracefully
        if not raw or raw.strip() in ('""', "''"):
            if verbose:
                print(f"[{context_name}] Empty response")
            return None

        def _sanitize_json_text(s: str) -> str:
            """Remove any leading non-json characters and fix common JSON issues"""
            # Remove any leading non-json characters before first { or [
            first = s.find("{")
            first_arr = s.find("[")
            if first == -1 or (first_arr != -1 and first_arr < first):
                first = first_arr
            if first > 0:
                s = s[first:]

            # Trim anything after the last closing brace/bracket
            last_obj = s.rfind("}")
            last_arr = s.rfind("]")
            last = max(last_obj, last_arr)
            if last != -1 and last < len(s) - 1:
                s = s[: last + 1]

            # Remove trailing commas before closing braces/brackets: {...,} or [...,]
            s = re.sub(r",\s*(?=[}\]])", "", s)

            # Remove accidental repeated commas
            s = re.sub(r",\s*,+", ",", s)

            return s.strip()

        def _escape_control_chars_in_json_strings(s: str) -> str:
            """Replace raw control characters inside JSON string literals with \\uXXXX escapes"""
            def _replace(match):
                txt = match.group(0)
                inner = txt[1:-1]
                new_inner = []
                i = 0
                while i < len(inner):
                    ch = inner[i]
                    # preserve existing escape sequences
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

        # Try direct JSON parse first (fast path)
        try:
            parsed = json.loads(raw)
            if verbose:
                print(f"[{context_name}] Direct JSON parse successful")
        except Exception as e:
            if verbose:
                print(f"[{context_name}] Direct parse failed: {e}")
            
            # Try sanitized variants
            candidate = None

            # Try to extract an outermost JSON array or object
            arr_match = re.search(r"(\[\s*[\s\S]*\])", raw)
            obj_match = re.search(r"(\{[\s\S]*\})", raw)
            if arr_match:
                candidate = arr_match.group(1)
            elif obj_match:
                candidate = obj_match.group(1)
            else:
                candidate = raw

            # sanitize candidate to fix trailing commas and stray text
            candidate = _sanitize_json_text(candidate)

            # escape control characters inside JSON strings
            candidate = _escape_control_chars_in_json_strings(candidate)

            if verbose:
                print(f"[{context_name}] Sanitized candidate: {candidate[:200]}...")

            # try parsing sanitized candidate
            try:
                parsed = json.loads(candidate)
                if verbose:
                    print(f"[{context_name}] Sanitized JSON parse successful")
            except Exception as e2:
                if verbose:
                    print(f"[{context_name}] Sanitized parse failed: {e2}")
                
                # last resort: try replacing single quotes with double then sanitize/escape again
                try:
                    cand2 = candidate.replace("'", '"')
                    cand2 = _sanitize_json_text(cand2)
                    cand2 = _escape_control_chars_in_json_strings(cand2)
                    parsed = json.loads(cand2)
                    if verbose:
                        print(f"[{context_name}] Quote-replaced JSON parse successful")
                except Exception as e3:
                    if verbose:
                        print(f"[{context_name}] All parsing attempts failed: {e3}")
                    parsed = None

        if parsed is None:
            if verbose:
                print(f"[{context_name}] Failed to parse as JSON after all fallbacks")
                print(f"[{context_name}] Raw response (truncated): {response_content[:500]}...")
            return None

        # Validate expected keys if provided
        if expected_keys and isinstance(parsed, dict):
            missing_keys = [key for key in expected_keys if key not in parsed]
            if missing_keys:
                if verbose:
                    print(f"[{context_name}] Warning: Missing expected keys: {missing_keys}")
                # Don't fail completely, just warn

        return parsed

    except Exception as e:
        if verbose:
            print(f"[{context_name}] Error parsing JSON: {e}")
        return None


def parse_ai_json_response_simple(response_content: str) -> Optional[Union[Dict, List]]:
    """
    Simplified version of parse_ai_json_response with minimal error handling.
    Use this for quick parsing when you're confident about the response format.
    
    Args:
        response_content: The raw response content from AI
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # Remove markdown code blocks
        raw = response_content.strip()
        if raw.startswith("```"):
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
        
        return json.loads(raw)
    except Exception:
        return None


def validate_json_structure(data: Union[Dict, List], required_keys: List[str] = None) -> bool:
    """
    Validate that a parsed JSON object has the required structure.
    
    Args:
        data: The parsed JSON data
        required_keys: List of keys that must be present if data is a dict
        
    Returns:
        True if structure is valid, False otherwise
    """
    if not data:
        return False
        
    if isinstance(data, dict) and required_keys:
        return all(key in data for key in required_keys)
        
    return True

