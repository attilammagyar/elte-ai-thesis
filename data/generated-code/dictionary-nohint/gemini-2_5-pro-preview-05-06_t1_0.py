# challenge='dictionary-nohint', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import collections.abc
import re
from collections import defaultdict

# Helper function to normalize dictionary keys and token text
def _normalize_text(text: str) -> str:
    """
    Normalizes text for dictionary keys and tokens.
    Converts to lowercase and canonicalizes word separation for phrases.
    E.g., "Black Swan" -> "black swan"
          "black,swan" -> "black swan"
          "blackswan"  -> "blackswan"
          "event."     -> "event"
          " "          -> ""
          "end-of-line"-> "end of line"
    """
    if not text:
        return ""
    lower_text = text.lower()
    # Split by any sequence of non-alphanumeric characters
    parts = re.split(r'[^a-zA-Z0-9]+', lower_text)
    # Filter out empty strings that can result from split
    # (e.g., from leading/trailing/consecutive separators)
    parts = [p for p in parts if p]
    
    if not parts:
        return ""
    
    # If multiple word components are found (e.g. "AAA,BBB" -> ["aaa","bbb"]),
    # they form a phrase joined by spaces.
    # If only one component (e.g. "AAABBB" -> ["aaabbb"] or "word." -> ["word"]),
    # it's a single word form.
    if len(parts) > 1:
        return " ".join(parts)
    else: # len(parts) == 1
        return parts[0]

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.
    """
    normalized_dictionary = defaultdict(set)
    for key_string, entry_id in dictionary.items():
        norm_key = _normalize_text(key_string)
        if norm_key:  # Ignore entries that normalize to empty string
            normalized_dictionary[norm_key].add(entry_id)
    return normalized_dictionary

def annotate(
    tokens: collections.abc.Iterable[str], 
    dictionary_index: object
) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.
    """
    # We expect dictionary_index to be the defaultdict returned by build_dictionary_index.
    # Explicit type casting for clarity / type checker support.
    idx: collections.abc.Mapping[str, set] = dictionary_index # type: ignore 

    tokens_list = list(tokens)
    if not tokens_list:
        return []

    n = len(tokens_list)
    # Initialize annotated_results as a list of tuples: (original_token_string, set_of_entry_ids)
    annotated_results: list[tuple[str, set]] = [(token, set()) for token in tokens_list]

    # Cache normalization of individual tokens as it's used frequently
    normalized_tokens_cache = [_normalize_text(t) for t in tokens_list]

    for i in range(n):  # Start index of the current segment tokens_list[i...j]
        # For Type 1 Match (Concatenated Words)
        current_concatenated_str_builder = []
        
        # For Type 2 Match (Phrases)
        current_phrase_word_parts = []
        # Store global indices in tokens_list for the first/last tokens contributing to current phrase
        current_phrase_first_word_token_idx = -1 
        current_phrase_last_word_token_idx = -1  

        for j in range(i, n):  # End index of the current segment tokens_list[i...j]
            token_text = tokens_list[j]
            token_original_idx = j # Global index of tokens_list[j]
            
            # Normalized form of the single token tokens_list[j], fetched from cache
            norm_current_token = normalized_tokens_cache[token_original_idx]

            # --- Type 1: Concatenated word match (e.g., "AAABBB", "blackswan") ---
            # Forms a candidate string by joining tokens_list[i...j].
            current_concatenated_str_builder.append(token_text)
            concatenated_segment_str = "".join(current_concatenated_str_builder)
            
            # Normalize the entire concatenated segment string.
            normalized_concatenated_key = _normalize_text(concatenated_segment_str)

            # Check if it's a "word" form (no spaces after normalization)
            if normalized_concatenated_key and ' ' not in normalized_concatenated_key:
                entry_ids = idx.get(normalized_concatenated_key, set())
                if entry_ids:
                    # Annotate all tokens within the segment tokens_list[i...j]
                    # that are not pure separators.
                    for k_token_idx in range(i, j + 1):
                        if normalized_tokens_cache[k_token_idx]: # Token is not a pure separator
                            annotated_results[k_token_idx][1].update(entry_ids)
            
            # --- Type 2: Phrase match (e.g., "AAA BBB", "black swan event") ---
            # Accumulate word parts from tokens in the segment tokens_list[i...j].
            if norm_current_token: # If current token (tokens_list[j]) contributes word material
                # A single token might normalize to multiple words (e.g. "end-of-line" -> "end of line")
                sub_parts = norm_current_token.split(' ') 
                current_phrase_word_parts.extend(sub_parts)
                
                # Update span of tokens that form this phrase
                if current_phrase_first_word_token_idx == -1: 
                    current_phrase_first_word_token_idx = token_original_idx
                current_phrase_last_word_token_idx = token_original_idx

            if current_phrase_word_parts: # If any word parts have been accumulated for a phrase
                normalized_phrase_key = " ".join(current_phrase_word_parts)
                entry_ids = idx.get(normalized_phrase_key, set())
                if entry_ids:
                    # Annotate tokens from the first word-contributing token up to the last
                    # word-contributing token (inclusive) that form this phrase.
                    # This correctly handles inner separators and leading/trailing separators.
                    if current_phrase_first_word_token_idx != -1: # Should be true if current_phrase_word_parts is non-empty
                        for k_token_idx in range(current_phrase_first_word_token_idx, current_phrase_last_word_token_idx + 1):
                            annotated_results[k_token_idx][1].update(entry_ids)
                            
    return annotated_results


# --- END GENERATED CODE ---

def test_empty_sentence():
    dictionary = {}
    tokens = []

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = []
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_token_not_in_dictionary():
    """
    When a token is not found in the dictionary
    then it should not be annotated with anything.
    """
    dictionary = {}
    tokens = ["AAA"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", set())]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_token_found_in_dictionary():
    """
    When a token is found in the dictionary
    then the token should be annotated with its dictionary entry.
    """
    dictionary = {"AAA": 1}
    tokens = ["AAA"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {1})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"


def test_case_insensitive_dictionary_lookup():
    """
    Dictionary lookup should be case-insensitive.
    """
    dictionary = {"AAA": 1, "BBB": 2}
    tokens = ["Aaa", "bbb"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("Aaa", {1}), ("bbb", {2})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_compound_phrase():
    """
    All tokens in a compound phrase which is found in the dictionary
    should be annotated with the dictionary entry of the phrase.
    """
    dictionary = {"AAA BBB": 1}
    tokens = ["AAA", " ", "BBB"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {1}), (" ", {1}), ("BBB", {1})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_compound_phrase_and_individual_word():
    """
    When a token is found in the dictionary both as an individual word and as part of a compound phrase
    then its annotations should include the dictionary entries of both the phrase and the individual word as well.
    """
    dictionary = {"AAA": 1, "BBB": 2, "AAA BBB": 3}
    tokens = ["AAA", " ", "BBB"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {3, 1}), (" ", {3}), ("BBB", {3, 2})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_compound_phrases_word_separation():
    """
    Compound phrase dictionary lookup should be insensitive to word separators.
    """
    dictionary = {"AAA": 1, "BBB": 2, "AAA, BBB": 3}
    tokens = ["AAA", " ", "*", "BBB", "*"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {3, 1}), (" ", {3}), ("*", {3}),("BBB", {3, 2}), ("*", set())]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_leading_and_trailing_separators_around_compound_phrase():
    """
    The leading and trailing word separators should not be considered
    parts of a compound phrase.
    """
    dictionary = {"AAA": 1, "BBB": 2, "AAA BBB": 3}
    tokens = [" ", "AAA", " ", "BBB", " "]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [(" ", set()), ("AAA", {3, 1}), (" ", {3}), ("BBB", {3, 2}), (" ", set())]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_separated_tokens_do_not_make_a_compound_word():
    """
    When tokens are separated by non-word characters
    then they should not be considered a compound word.
    """
    dictionary = {"AAA": 1, "BBB": 2, "AAABBB": 3}
    tokens = ["AAA", " ", "BBB"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {1}), (" ", set()), ("BBB", {2})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_compound_word_tokens_missing_from_dictionary():
    """
    Compound words may contain tokens which are not listed in the dictionary
    as individual words.
    """
    dictionary = {"AAABBB": 1}
    tokens = ["AAA", "BBB"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", {1}), ("BBB", {1})]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_compound_phrase_overlap():
    """
    Tokens in overlapping compound phrases should be annotated with the
    dictionary entries for all compound phrases in which they participate.
    """
    dictionary = {
        "AAA": 1,
        "BBB": 2,
        "CCC": 3,
        "AAA BBB": 4,
        "BBB CCC": 5,
        "CCC CCC": 6,
    }
    tokens = ["AAA", " ", "BBB", " ", "CCC", " ", "CCC", " ", "CCC"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [
        ("AAA", {4, 1}),
        (" ", {4}),
        ("BBB", {5, 4, 2}),
        (" ", {5}),
        ("CCC", {6, 5, 3}),
        (" ", {6}),
        ("CCC", {6, 3}),
        (" ", {6}),
        ("CCC", {6, 3}),
    ]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_nested_compound_phrases():
    """
    When a compound phrase itself is a part of a larger compound phrase
    then its tokens should be annotated with the dictionary entries for all the nested compound phrases.
    """
    dictionary = {
        "AAA": 1,
        "BBB": 2,
        "CCC": 3,
        "DDD": 4,
        "EEE": 5,
        "AAA BBB CCC DDD EEE": 6,
        "BBB CCC DDD": 7,
        "BBB CCC": 8,
    }
    tokens = ["AAA", " ", "BBB", " ", "CCC", " ", "DDD", " ", "EEE"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [
        ("AAA", {6, 1}),
        (" ", {6}),
        ("BBB", {6, 7, 8, 2}),
        (" ", {6, 7, 8}),
        ("CCC", {6, 7, 8, 3}),
        (" ", {6, 7}),
        ("DDD", {6, 7, 4}),
        (" ", {6}),
        ("EEE", {6, 5}),
    ]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_no_midtoken_match():
    """
    Dictionary entry match must occur at token end.
    """
    dictionary = {"AA": 1, "AAA BBB": 2, "CC": 3, "CCCDDD": 4}
    tokens = ["AAA", " ", "BBBCCC", "CCC", "DDDEEE"]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [("AAA", set()), (" ", set()), ("BBBCCC", set()), ("CCC", set()), ("DDDEEE", set())]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def test_real_life_example():
    dictionary = {
        "a": 1,
        "black": 2,
        "swan": 3,
        "black swan": 4,
        "event": 5,
        "black swan event": 6,
        "would": 7,
        "occur": 8,
        "less": 9,
        "than": 10,
        "once": 11,
        "in": 12,
        "blue": 13,
        "moon": 14,
        "blue moon": 15,
        "once in a blue moon": 16,
    }
    tokens = [
        "A", " ", "black", " ", "swan", " ", "event", " ", "would", " ",
        "occur", " ", "less", " ", "than", " ", "once", " ", "in", " ", "a",
        " ", "blue", " ", "moon", ".",
    ]

    dictionary_index = build_dictionary_index(dictionary)
    annotated_tokens = list(annotate(tokens, dictionary_index))

    expected = [
        ("A", {1}),
        (" ", set()),
        ("black", {2, 4, 6}),
        (" ", {4, 6}),
        ("swan", {3, 4, 6}),
        (" ", {6}),
        ("event", {5, 6}),
        (" ", set()),
        ("would", {7}),
        (" ", set()),
        ("occur", {8}),
        (" ", set()),
        ("less", {9}),
        (" ", set()),
        ("than", {10}),
        (" ", set()),
        ("once", {11, 16}),
        (" ", {16}),
        ("in", {12, 16}),
        (" ", {16}),
        ("a", {1, 16}),
        (" ", {16}),
        ("blue", {13, 15, 16}),
        (" ", {15, 16}),
        ("moon", {14, 15, 16}),
        (".", set()),
    ]
    assert annotated_tokens == expected, f"{expected=}, {annotated_tokens=}"

def perf_test():
    import random
    import time

    dictionary = {
        "a": 1,
        "black": 2,
        "swan": 3,
        "black swan": 4,
        "event": 5,
        "black swan event": 6,
        "would": 7,
        "occur": 8,
        "less": 9,
        "than": 10,
        "once": 11,
        "in": 12,
        "blue": 13,
        "moon": 14,
        "blue moon": 15,
        "once in a blue moon": 16,
    }
    tokens = [
        "A", " ", "black", " ", "swan", " ", "event", " ", "would", " ",
        "occur", " ", "less", " ", "than", " ", "once", " ", "in", " ", "a",
        " ", "blue", " ", "moon", ".",
    ]
    letters = "abcdefghijklmnopqrstuvwxyz"

    while len(dictionary) < 1000:
        random_word = "".join([random.choice(letters) for i in range(15)])
        random_expr = (
            "".join([random.choice(letters) for i in range(7)])
            + " "
            + "".join([random.choice(letters) for i in range(7)])
        )
        dictionary[random_word] = len(dictionary)
        dictionary[random_expr] = len(dictionary)

    for i in range(6):
        tokens = tokens + tokens

    begin = time.time()

    for i in range(100):
        tokens_copy = list(tokens)
        dictionary_copy = dict(dictionary)
        dictionary_index = build_dictionary_index(dictionary_copy)
        annotated_tokens = list(annotate(tokens_copy, dictionary_index))

    end = time.time()

    return end - begin


def run_tests():
    import json
    import sys

    module = sys.modules[__name__]

    tests = []

    for name, value in globals().items():
        if name.startswith("test_") and callable(value) and value.__code__.co_argcount == 0:
            tests.append(value)

    passed = 0
    failed = 0
    failures = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            failed += 1
            failures.append(f"{type(exc)} {exc}")

    real_life_test_passed = 0

    try:
        test_real_life_example()
        real_life_test_passed = 1
    except:
        pass
        

    perf = perf_test()

    results = {
        "passed": passed,
        "failed": failed,
        "perf": perf,
        "failures": failures,
        "real_life_test_passed": real_life_test_passed,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_tests()

