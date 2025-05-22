# challenge='dictionary-hint', generator='gemini-2.5-pro-preview-05-06', temperature=0.3
# --- BEGIN GENERATED CODE ---

import collections.abc

class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.entry_ids: set = set()

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.

    Parameters:
        dictionary: Mapping strings (keys) to meanings (values).
    """
    root = TrieNode()

    for key_str, entry_id in dictionary.items():
        node = root
        key_lower = key_str.lower()

        if not key_lower:  # Empty string key
            root.entry_ids.add(entry_id)
            continue

        # current_path_ends_in_space tracks if the path *built so far for the current key*
        # ends in a normalized space character in the Trie.
        current_path_ends_in_space = False 

        for char in key_lower:
            if char.isalnum():
                node = node.children.setdefault(char, TrieNode())
                current_path_ends_in_space = False
            else:  # char is a separator
                if not current_path_ends_in_space:
                    node = node.children.setdefault(' ', TrieNode())
                    current_path_ends_in_space = True
                # If current_path_ends_in_space is already True, we absorb the separator
                # and don't add another space to the trie path for this specific char.
        
        node.entry_ids.add(entry_id)
        
    return root

# Helper function to check if a token is a separator
def _is_separator_token(token_str: str) -> bool:
    if not token_str: # Treat empty string as separator, though tokens are usually non-empty
        return True
    return not any(c.isalnum() for c in token_str)

# Helper function to filter leading/trailing separators for annotation
def _filter_indices_for_annotation(
    token_indices: list[int],
    all_tokens_list: list[str]
) -> list[int]:
    if not token_indices:
        return []

    start_ptr = 0
    while start_ptr < len(token_indices) and \
          _is_separator_token(all_tokens_list[token_indices[start_ptr]]):
        start_ptr += 1
    
    end_ptr = len(token_indices) - 1
    while end_ptr >= start_ptr and \
          _is_separator_token(all_tokens_list[token_indices[end_ptr]]):
        end_ptr -= 1
        
    if start_ptr > end_ptr: # All tokens in match were separators
        return []
    
    # Create a new list for the slice
    return list(token_indices[start_ptr : end_ptr + 1])


# Helper function to add annotations
def _add_annotations_to_results(
    annotated_tokens_list: list[tuple[str, set]],
    entry_ids_to_add: set,
    list_of_all_token_indices_in_match: list[int],
    all_tokens_list: list[str]
):
    actual_indices_to_annotate = _filter_indices_for_annotation(
        list_of_all_token_indices_in_match,
        all_tokens_list
    )
    for token_idx_to_annotate in actual_indices_to_annotate:
        annotated_tokens_list[token_idx_to_annotate][1].update(entry_ids_to_add)


def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    trie_root: TrieNode = dictionary_index # type: ignore 
    tokens_list = list(tokens) # Ensure indexable list
    
    if not tokens_list:
        return []

    annotated_results = [(token, set()) for token in tokens_list]

    # Candidate: (current_trie_node, start_token_idx_of_match, list_of_contributing_token_indices, path_in_trie_ended_with_space)
    active_candidates: list[tuple[TrieNode, int, list[int], bool]] = []

    for i, current_token_original in enumerate(tokens_list):
        current_token_lower = current_token_original.lower()
        
        next_active_candidates: list[tuple[TrieNode, int, list[int], bool]] = []

        # 1. Advance existing candidates
        for cand_node, cand_start_idx, cand_token_indices, cand_path_ended_space in active_candidates:
            current_trie_path_node = cand_node
            
            if _is_separator_token(current_token_original):
                if not cand_path_ended_space: 
                    if ' ' in current_trie_path_node.children:
                        advanced_node = current_trie_path_node.children[' ']
                        new_contrib_indices = cand_token_indices + [i]
                        next_active_candidates.append((advanced_node, cand_start_idx, new_contrib_indices, True))
                        if advanced_node.entry_ids:
                            _add_annotations_to_results(annotated_results, advanced_node.entry_ids, new_contrib_indices, tokens_list)
                else: 
                    new_contrib_indices = cand_token_indices + [i]
                    next_active_candidates.append((current_trie_path_node, cand_start_idx, new_contrib_indices, True)) 
                    if current_trie_path_node.entry_ids:
                         _add_annotations_to_results(annotated_results, current_trie_path_node.entry_ids, new_contrib_indices, tokens_list)
            
            else: # Current token is a word/word-part
                temp_node_for_traversal = current_trie_path_node
                possible_to_traverse_token = True
                processed_any_alnum_in_token = False

                for char_in_token in current_token_lower:
                    if char_in_token.isalnum():
                        if char_in_token in temp_node_for_traversal.children:
                            temp_node_for_traversal = temp_node_for_traversal.children[char_in_token]
                            processed_any_alnum_in_token = True
                        else:
                            possible_to_traverse_token = False
                            break
                    # Non-alnum chars in a "word" token (e.g. '.' in "word.") are skipped for matching.
                
                if possible_to_traverse_token and processed_any_alnum_in_token:
                    new_contrib_indices = cand_token_indices + [i]
                    next_active_candidates.append((temp_node_for_traversal, cand_start_idx, new_contrib_indices, False))
                    if temp_node_for_traversal.entry_ids:
                        _add_annotations_to_results(annotated_results, temp_node_for_traversal.entry_ids, new_contrib_indices, tokens_list)

        # 2. Start new candidates from trie_root
        current_trie_path_node = trie_root
        
        if _is_separator_token(current_token_original):
            if ' ' in current_trie_path_node.children: 
                started_node = current_trie_path_node.children[' ']
                next_active_candidates.append((started_node, i, [i], True))
                if started_node.entry_ids:
                    _add_annotations_to_results(annotated_results, started_node.entry_ids, [i], tokens_list)
        else: # Current token is a word/word-part
            temp_node_for_traversal = current_trie_path_node
            possible_to_traverse_token = True
            processed_any_alnum_in_token = False
            for char_in_token in current_token_lower:
                if char_in_token.isalnum():
                    if char_in_token in temp_node_for_traversal.children:
                        temp_node_for_traversal = temp_node_for_traversal.children[char_in_token]
                        processed_any_alnum_in_token = True
                    else:
                        possible_to_traverse_token = False
                        break
            
            if possible_to_traverse_token and processed_any_alnum_in_token:
                next_active_candidates.append((temp_node_for_traversal, i, [i], False))
                if temp_node_for_traversal.entry_ids: 
                    _add_annotations_to_results(annotated_results, temp_node_for_traversal.entry_ids, [i], tokens_list)
        
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

