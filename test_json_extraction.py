#!/usr/bin/env python3
"""
Unit tests for the extract_json_from_generation function bug fix.

This test demonstrates the bug where the greedy regex pattern r'\{.*\}'
would match from the first '{' to the LAST '}' in text, instead of
extracting the first complete JSON object.
"""

import json
import unittest
import sys
import os

# Add parent directory to path to import test.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test import extract_json_from_generation


class TestExtractJsonFromGeneration(unittest.TestCase):
    """Test cases for extract_json_from_generation function."""
    
    def test_single_json_object(self):
        """Test extraction of a single JSON object."""
        text = '{"table": "trades", "filters": []}'
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
        self.assertIsInstance(parsed["filters"], list)
    
    def test_json_with_nested_objects(self):
        """Test extraction of JSON with nested objects."""
        text = '{"table": "trades", "filters": [{"column": "id", "op": "=", "value": 123}]}'
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
        self.assertEqual(len(parsed["filters"]), 1)
    
    def test_json_followed_by_extra_text(self):
        """
        Test extraction when JSON is followed by extra text.
        
        This is the main bug case: when the model generates valid JSON
        followed by extra tokens, the greedy regex would fail to extract
        just the first complete JSON object.
        """
        text = '{"table": "trades", "filters": []} extra text here'
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result, "Should extract JSON even with trailing text")
        
        # Should be able to parse the extracted JSON
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
        self.assertIsInstance(parsed["filters"], list)
    
    def test_multiple_json_objects(self):
        """
        Test extraction when there are multiple JSON objects.
        
        Critical bug case: the greedy regex r'\{.*\}' would match from
        the first '{' to the LAST '}', creating invalid JSON.
        
        For example, with input:
        '{"table": "a", "filters": []} {"table": "b", "filters": []}'
        
        The greedy regex would return:
        '{"table": "a", "filters": []} {"table": "b", "filters": []}'
        
        Which is invalid JSON (two objects concatenated).
        
        The fix should return only the first complete JSON object:
        '{"table": "a", "filters": []}'
        """
        text = '{"table": "trades", "filters": []} {"table": "other", "filters": []}'
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result, "Should extract first JSON object")
        
        # Should be able to parse the extracted JSON
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
        
        # Should NOT contain the second object
        self.assertNotIn("other", result)
    
    def test_json_with_special_tokens(self):
        """Test extraction with Llama-3 special tokens."""
        text = '<|eot_id|>{"table": "trades", "filters": []}<|end_header_id|>'
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
    
    def test_json_with_trailing_incomplete_json(self):
        """
        Test the realistic case from test_results.json.
        
        The model generates valid JSON followed by incomplete JSON,
        which was causing the greedy regex to match too much.
        """
        text = '''{\n  "table": "trades",\n  "filters": [\n    {\n      "column": "term_code",\n      "op": "LIKE",\n      "value": "TEST"\n    }\n  ]\n}<|reserved_special_token_23|><|reserved_special_token_39|>assistant<|reserved_special_token_128|>\n\n{\n  "table": "trades",\n  "filters": [\n    {\n      "column": "term_code",\n      "op": "LIKE",\n'''
        
        result = extract_json_from_generation(text)
        self.assertIsNotNone(result, "Should extract first complete JSON")
        
        # Should be able to parse the extracted JSON
        parsed = json.loads(result)
        self.assertEqual(parsed["table"], "trades")
        self.assertEqual(len(parsed["filters"]), 1)
        self.assertEqual(parsed["filters"][0]["column"], "term_code")
    
    def test_no_json_in_text(self):
        """Test that None is returned when no JSON is present."""
        text = "This is just plain text without any JSON"
        result = extract_json_from_generation(text)
        self.assertIsNone(result)
    
    def test_incomplete_json(self):
        """Test that None is returned for incomplete JSON."""
        text = '{"table": "trades", "filters": ['
        result = extract_json_from_generation(text)
        self.assertIsNone(result, "Should return None for incomplete JSON")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
