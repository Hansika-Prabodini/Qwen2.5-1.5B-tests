# Bug Fix Summary: JSON Extraction in extract_json_from_generation()

## Bug Description

The `extract_json_from_generation()` function in both `test.py` (line 221) and `inference.py` (line 192) contained a critical bug in the regular expression pattern used to extract JSON objects from generated text.

**Buggy Code:**
```python
json_match = re.search(r'\{.*\}', text, re.DOTALL)
```

### The Problem

The regex pattern `r'\{.*\}'` with the `re.DOTALL` flag is **greedy**, meaning `.*` matches as much text as possible. This causes the pattern to match from the **first** `{` to the **LAST** `}` in the text, rather than extracting just the first complete JSON object.

### Impact

Based on `test_results.json`, this bug caused **100% failure rate** (0/150 test examples passing) in JSON parsing, even though the model was generating valid JSON. The bug manifested when:

1. The model generated multiple JSON objects
2. The model generated valid JSON followed by extra text or tokens
3. The text contained nested JSON structures followed by incomplete JSON

**Example from test_results.json (example_id: 0):**
```
Generated text: {"table": "trades", "filters": [...]}
<|reserved_special_token_23|>...
{"table": "trades", "filters": [...
```

The greedy regex would match from the first `{` all the way to a `}` in the incomplete second JSON object, resulting in malformed JSON that couldn't be parsed.

## The Fix

Replaced the greedy regex with proper brace-counting logic that extracts the first complete JSON object:

```python
# Find the first '{' character
start_idx = text.find('{')
if start_idx == -1:
    return None

# Count braces to find the matching closing brace
brace_count = 0
for i in range(start_idx, len(text)):
    if text[i] == '{':
        brace_count += 1
    elif text[i] == '}':
        brace_count -= 1
        if brace_count == 0:
            # Found the matching closing brace
            return text[start_idx:i+1]

# No matching closing brace found
return None
```

### Why This Fix Works

1. **Finds the first `{`**: Locates the start of the JSON object
2. **Counts braces**: Properly tracks nesting by incrementing for `{` and decrementing for `}`
3. **Stops at matching `}`**: Returns immediately when brace_count reaches 0, extracting only the first complete JSON object
4. **Handles nested objects**: Correctly handles nested `{}` structures
5. **Returns None for incomplete JSON**: If no matching `}` is found, returns None

## Files Modified

1. `test.py` - Fixed `extract_json_from_generation()` at line 201-225
2. `inference.py` - Fixed `extract_json_from_generation()` at line 172-196

## Unit Test

Created `test_json_extraction.py` with comprehensive test cases that:

- **Would FAIL before the patch** (especially `test_multiple_json_objects` and `test_json_with_trailing_incomplete_json`)
- **PASS after the patch**

Key test cases:
- Single JSON object extraction
- JSON with nested objects
- JSON followed by extra text
- **Multiple JSON objects** (critical bug case)
- **JSON with trailing incomplete JSON** (realistic case from test_results.json)
- JSON with special tokens
- No JSON in text
- Incomplete JSON

## Expected Impact

This fix should dramatically improve the parse success rate in `test.py` evaluation. The test_results.json showed 0% parse success rate (0/150 examples), which should increase significantly as the function can now correctly extract valid JSON from the model's generated output.

## Verification

To verify the fix:

```bash
# Run the unit tests
python test_json_extraction.py

# Re-run the evaluation (if model and test data are available)
# python test.py --model <model_path> --test-data <test_data_path>
```

## Root Cause Analysis

The root cause was using a greedy regex quantifier (`.*`) for pattern matching in text that commonly contains multiple `}` characters. Regular expressions are not well-suited for matching balanced constructs like JSON objects with nested braces. The proper solution is to use a brace-counting algorithm, which is exactly what the fix implements.
