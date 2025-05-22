# challenge='dictionary-hint', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import collections.abc

# Canonical separator used in the Trie for normalized phrase keys.
_SEP_ = ' '

class TrieNode:
    """Node for the Trie data structure."""
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        # Stores dictionary entry IDs for words/phrases ending at this node.
        self.entry_ids: set = set()

def _is_separator(token: str) -> bool:
    """
    Determines if a token is a separator.
    A token is considered a separator if it contains no alphanumeric characters.
    Examples: " ", "*", ".", "" are separators. "word", "w1" are not.
    """
    return not any(c.isalnum() for c in token)

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.

    The index is a Trie. Keys are normalized:
    1. Converted to lowercase.
    2. Sequences of separator characters are replaced by a single canonical separator `_SEP_`.
    3. Leading/trailing separators are removed from the key.
       Example: "  Word1,  Word2 " -> "word1<_SEP_>word2"
    4. Pure separator keys are ignored.

    Parameters:
        dictionary: Mapping strings (keys) to meanings (values/entry_ids).
    Returns:
        The root TrieNode of the built index.
    """
    root = TrieNode()
    
    for key, entry_id in dictionary.items():
        normalized_key_parts = []
        current_word_part = ""
        key_lower = key.lower()

        for char_in_key in key_lower:
            if char_in_key.isalnum():
                current_word_part += char_in_key
            else: # Character is a separator type
                if current_word_part:
                    normalized_key_parts.append(current_word_part)
                    current_word_part = ""
                # Add canonical separator if list is empty or last part isn't already _SEP_
                if not normalized_key_parts or normalized_key_parts[-1] != _SEP_:
                    normalized_key_parts.append(_SEP_)
        
        if current_word_part: # Append any trailing word part
            normalized_key_parts.append(current_word_part)

        # Remove leading/trailing _SEP_ from the normalized parts
        start_idx = 0
        while start_idx < len(normalized_key_parts) and normalized_key_parts[start_idx] == _SEP_:
            start_idx += 1
        
        end_idx = len(normalized_key_parts) - 1
        while end_idx >= start_idx and normalized_key_parts[end_idx] == _SEP_:
            end_idx -= 1
        
        final_normalized_parts = normalized_key_parts[start_idx : end_idx + 1]

        if not final_normalized_parts: # Key was empty or consisted only of separators
            continue
            
        trie_key_string = "".join(final_normalized_parts)

        node = root
        for char_in_trie_key in trie_key_string:
            node = node.children.setdefault(char_in_trie_key, TrieNode())
        node.entry_ids.add(entry_id)
        
    return root

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.

    Parameters:
        dictionary_index:   A dictionary index (TrieNode root) from build_dictionary_index().
        tokens:             The tokens to be annotated.

    Return:
        annotated_tokens:   A list containing (token, annotations) pairs.
    """
    token_list = list(tokens) 
    if not token_list:
        return []

    root_node = dictionary_index # Type hint could be TrieNode if not for 'object'

    annotated_results = [(token, set()) for token in token_list]

    # active_candidates: list of (current_trie_node, start_token_idx, prev_token_in_match_was_separator)
    active_candidates: list[tuple[TrieNode, int, bool]] = []

    for i, current_token_str in enumerate(token_list):
        current_token_lower = current_token_str.lower()
        is_current_token_separator = _is_separator(current_token_str)
        
        next_active_candidates: list[tuple[TrieNode, int, bool]] = []

        # 1. Start new candidates with current_token_str.
        if not is_current_token_separator:
            node = root_node
            path_valid = True
            for char_val in current_token_lower:
                if char_val in node.children:
                    node = node.children[char_val]
                else:
                    path_valid = False
                    break
            
            if path_valid:
                candidate_tuple = (node, i, False) # Current token is not a separator
                next_active_candidates.append(candidate_tuple)

                if node.entry_ids:
                    for entry_id in node.entry_ids:
                        annotated_results[i][1].add(entry_id)
        
        # 2. Extend existing candidates with current_token_str.
        for prev_trie_node, match_start_idx, prev_token_was_separator in active_candidates:
            current_path_node = prev_trie_node
            transition_made = True

            if is_current_token_separator:
                if not prev_token_was_separator: # Previous token was a word
                    if _SEP_ in current_path_node.children:
                        current_path_node = current_path_node.children[_SEP_]
                    else:
                        transition_made = False
                # Else (prev_token_was_separator is True): current separator is absorbed, node doesn't change.
            else: # Current token is a word/word-piece.
                for char_val in current_token_lower:
                    if char_val in current_path_node.children:
                        current_path_node = current_path_node.children[char_val]
                    else:
                        transition_made = False
                        break
            
            if transition_made:
                candidate_tuple = (current_path_node, match_start_idx, is_current_token_separator)
                next_active_candidates.append(candidate_tuple)

                if current_path_node.entry_ids:
                    first_word_token_idx_in_match = -1
                    last_word_token_idx_in_match = -1
                    
                    for k in range(match_start_idx, i + 1):
                        if not _is_separator(token_list[k]):
                            if first_word_token_idx_in_match == -1:
                                first_word_token_idx_in_match = k
                            last_word_token_idx_in_match = k
                    
                    if first_word_token_idx_in_match != -1:
                        for entry_id in current_path_node.entry_ids:
                            for k_anno_idx in range(first_word_token_idx_in_match, last_word_token_idx_in_match + 1):
                                annotated_results[k_anno_idx][1].add(entry_id)
        
        active_candidates = next_active_candidates
        
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

