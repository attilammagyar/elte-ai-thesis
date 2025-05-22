# challenge='dictionary-nohint', generator='gemini-2.5-pro-preview-05-06', temperature=0.5
# --- BEGIN GENERATED CODE ---

import collections.abc
import re

# Helper functions
def _is_word_token(token: str) -> bool:
    """Checks if a token is a word token (contains at least one alphanumeric character)."""
    return any(c.isalnum() for c in token)

def _is_separator_token(token: str) -> bool:
    """Checks if a token is a separator token (contains no alphanumeric characters)."""
    # A token is a separator if it's not a word token.
    # This definition implies tokens like "word-word" are word tokens if '-' is not alphanumeric.
    return not _is_word_token(token)

class DictionaryIndex:
    """
    Wrapper class for the dictionary index structure.
    The index itself is a dict mapping lookup keys (strings or tuples)
    to lists of (entry_id, is_original_entry_a_phrase) tuples.
    """
    def __init__(self):
        self.index = collections.defaultdict(list)

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.
    """
    idx_obj = DictionaryIndex()

    for key_str, entry_id in dictionary.items():
        key_lower = key_str.lower()
        
        word_parts = re.findall(r'[a-zA-Z0-9]+', key_lower)

        if not word_parts:
            # This case implies the dictionary key itself might be purely a separator string (e.g., ",-,").
            # Such entries are typically not expected or are handled by context.
            # For this problem, we assume dictionary keys usually contain word content.
            # If a key like " , " needs to be matched, its handling would depend on specific rules
            # for purely separator phrases, which are not detailed in the requirements.
            # We will proceed assuming word_parts will be non-empty for meaningful entries.
            # If it can be empty and needs specific handling, requirements would need to clarify.
             pass # Continue, or handle as a special type of entry if required.


        # is_original_entry_a_phrase: True if the dictionary key string contains separators
        # between its word components, or if it's a single word part but the key
        # had surrounding/trailing non-alphanumeric characters (e.g., "word.").
        # If word_parts is empty (e.g. key_str = "---"), then "".join(word_parts) is "".
        # len("") < len("---") is True. So "---" would be considered a phrase.
        is_original_entry_a_phrase = (len("".join(word_parts)) < len(key_lower)) if word_parts else True


        # 1. Store for exact match lookup (key is the lowercased original string)
        # Handles: "word", "compoundword", "phrase with specific separators"
        idx_obj.index[key_lower].append((entry_id, is_original_entry_a_phrase))
        
        # 2. Store for separator-insensitive phrase lookup (key is a tuple of word parts)
        # This is only added if the original entry was structured as a phrase.
        if is_original_entry_a_phrase and word_parts:
            # Example: "AAA, BBB" (phrase) -> lookup key ("aaa", "bbb")
            # Example: "word." (phrase) -> lookup key ("word",)
            phrase_lookup_key = tuple(word_parts)
            # Add this entry, ensuring the `is_original_entry_a_phrase` flag is True.
            # Note: A dictionary key might be added to phrase_lookup_key path
            # only if it was originally a phrase. The flag is always True here.
            idx_obj.index[phrase_lookup_key].append((entry_id, True))
            
    return idx_obj

def annotate(tokens: collections.abc.Iterable[str], dictionary_index_obj: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.
    """
    if not isinstance(dictionary_index_obj, DictionaryIndex):
        # This indicates incorrect usage or an unexpected type for dictionary_index_obj.
        # Depending on desired robustness, could raise TypeError or handle differently.
        # For the scope of this problem, we assume it's the correct object.
        raise TypeError("dictionary_index_obj is not of type DictionaryIndex")

    idx = dictionary_index_obj.index
    
    tokens_list = list(tokens) # Ensure it's a list for indexed access
    n = len(tokens_list)
    annotated_tokens_result = [(token, set()) for token in tokens_list]

    if n == 0:
        return []

    for i in range(n):  # Start index of the token subsequence
        current_concatenated_str = ""
        current_word_sequence_from_tokens = [] # Stores lowercase word strings from tokens[i...j]
        
        for j in range(i, n):  # End index of the token subsequence (span tokens_list[i...j])
            token_j_str = tokens_list[j]
            
            # --- Part 1: Exact concatenated string match ---
            current_concatenated_str += token_j_str
            lookup_key1 = current_concatenated_str.lower()

            if lookup_key1 in idx:
                for entry_id, is_orig_phrase_flag in idx[lookup_key1]:
                    annotate_start_idx, annotate_end_idx = i, j
                    if is_orig_phrase_flag: # Rule applies if the matched dictionary entry was a phrase
                        if _is_separator_token(tokens_list[annotate_start_idx]):
                            annotate_start_idx += 1
                        # Ensure annotate_start_idx does not cross annotate_end_idx after incrementing
                        if annotate_start_idx <= annotate_end_idx and \
                           _is_separator_token(tokens_list[annotate_end_idx]):
                            annotate_end_idx -= 1
                    
                    if annotate_start_idx <= annotate_end_idx: # Ensure valid range
                        for k_idx in range(annotate_start_idx, annotate_end_idx + 1):
                            annotated_tokens_result[k_idx][1].add(entry_id)

            # --- Part 2: Separator-insensitive phrase match ---
            if _is_word_token(token_j_str):
                current_word_sequence_from_tokens.append(token_j_str.lower())

            if not current_word_sequence_from_tokens: # If span i..j has no word tokens yet
                continue

            lookup_key2 = tuple(current_word_sequence_from_tokens)
            if lookup_key2 in idx:
                # This key (tuple of words) implies a phrase structure.
                # Check if the current token span tokens_list[i...j] supports this structure.
                # If lookup_key2 has multiple words, tokens_list[i...j] must contain separators.
                num_words_in_token_span = len(current_word_sequence_from_tokens)
                num_total_tokens_in_span = (j - i + 1)
                
                is_valid_token_phrase_structure = True
                if num_words_in_token_span > 1: # Multi-word phrase from tokens
                    # If all tokens in the span were words, it's not a phrase with separators.
                    if num_total_tokens_in_span == num_words_in_token_span:
                        is_valid_token_phrase_structure = False
                # If num_words_in_token_span == 1, it's structurally valid (e.g. matching "word.")
                
                if is_valid_token_phrase_structure:
                    for entry_id, is_orig_phrase_flag in idx[lookup_key2]:
                        # This type of match (via tuple of words) is only for dictionary entries
                        # that were themselves phrases.
                        if not is_orig_phrase_flag:
                            continue
                        
                        annotate_start_idx, annotate_end_idx = i, j
                        # Since is_orig_phrase_flag is True, apply leading/trailing separator rule.
                        if _is_separator_token(tokens_list[annotate_start_idx]):
                            annotate_start_idx += 1
                        if annotate_start_idx <= annotate_end_idx and \
                           _is_separator_token(tokens_list[annotate_end_idx]):
                            annotate_end_idx -= 1

                        if annotate_start_idx <= annotate_end_idx: # Ensure valid range
                            for k_idx in range(annotate_start_idx, annotate_end_idx + 1):
                                annotated_tokens_result[k_idx][1].add(entry_id)
                                
    return annotated_tokens_result



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

