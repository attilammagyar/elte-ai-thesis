# challenge='dictionary-nohint', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import collections.abc

# Helper class for the Trie
class _TrieNode:
    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.values: set = set() # Stores dictionary entry IDs
        self.is_end_of_entry: bool = False

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.

    Parameters:
        dictionary: Mapping strings (keys) to meanings (values).
    """
    root = _TrieNode()

    for original_key, value in dictionary.items():
        # Normalize the key
        processed_key = original_key.lower()
        
        normalized_key_for_trie = ""
        # True if the previous character processed was a separator, or at the start.
        # Used to collapse multiple separators into one space and handle leading/trailing.
        prev_char_was_separator = True 
                                     
        for char_code in processed_key:
            if char_code.isalnum():
                normalized_key_for_trie += char_code
                prev_char_was_separator = False
            else: # Separator character
                if not prev_char_was_separator: # Add space only if previous wasn't separator
                    normalized_key_for_trie += ' '
                prev_char_was_separator = True # Mark that we've seen a separator
        
        # Remove leading/trailing spaces that might result from normalization.
        # E.g., if original key was " foo " or " ,, foo ,, ".
        normalized_key_for_trie = normalized_key_for_trie.strip(' ')
        
        # If, after normalization, the key is empty (e.g. dictionary key was " " or ",,"), skip it.
        if not normalized_key_for_trie:
            continue

        # Insert into Trie
        node = root
        for char_code in normalized_key_for_trie:
            node = node.children.setdefault(char_code, _TrieNode())
        
        node.values.add(value)
        node.is_end_of_entry = True
        
    return root

def _try_extend_trie_path_with_token(
    token_str: str,
    is_path_empty: bool,
    path_ends_with_space: bool, # Relevant only if not is_path_empty
    current_trie_node: _TrieNode
) -> tuple[str, _TrieNode | None, bool]:
    """
    Tries to extend the current Trie path with the given token.

    Returns:
        A tuple (characters_added_to_path, new_trie_node, success_flag).
        `characters_added_to_path`: The string segment that this token contributed to the Trie path.
        `new_trie_node`: The Trie node reached after processing this token. None if path broken.
        `success_flag`: True if the path could be extended (or token processed neutrally), False otherwise.
    """
    token_lower = token_str.lower()
    
    next_node_candidate = current_trie_node
    chars_for_path_extension = ""

    if token_str.isalnum(): # Token is a word or word-piece
        temp_node = next_node_candidate
        for char_code in token_lower:
            if char_code in temp_node.children:
                temp_node = temp_node.children[char_code]
                chars_for_path_extension += char_code
            else:
                return "", None, False # Path broken
        next_node_candidate = temp_node
        return chars_for_path_extension, next_node_candidate, True
    
    else: # Token is a separator
        if is_path_empty:
            # Cannot start a Trie match with a separator token, as Trie keys are trimmed.
            return "", None, False

        if path_ends_with_space:
            # Path already ends with a space; this separator is redundant for lookup.
            # No change to path string or Trie node, but token is "processed".
            return "", next_node_candidate, True # chars_for_path_extension is empty
        else:
            # Path does not end with a space; add one for this separator.
            if ' ' in next_node_candidate.children:
                next_node_candidate = next_node_candidate.children[' ']
                chars_for_path_extension = ' '
                return chars_for_path_extension, next_node_candidate, True
            else:
                # Trie path expected an alphanumeric char here, but got a separator (space).
                return "", None, False

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.
    """
    trie_root: _TrieNode = dictionary_index # type: ignore
    
    tokens_list = list(tokens)
    if not tokens_list:
        return []

    n = len(tokens_list)
    annotated_results = [(token, set()) for token in tokens_list]

    for i in range(n): # Starting token index of a potential match
        current_node_in_trie: _TrieNode = trie_root
        
        # State of the conceptual path string formed by tokens[i...j]
        is_current_path_empty = True
        current_path_ends_with_space = False # Irrelevant if path is empty
        
        first_word_token_idx_in_span = -1 # Index of the first word-like token in tokens[i...j]

        for j in range(i, n): # Ending token index of the potential match
            current_token = tokens_list[j]
            
            chars_added_by_token, next_node_candidate, success = _try_extend_trie_path_with_token(
                current_token, is_current_path_empty, current_path_ends_with_space, current_node_in_trie
            )

            if success:
                current_node_in_trie = next_node_candidate # type: ignore
                
                # Update path state
                is_current_path_empty = False
                if chars_added_by_token: 
                    current_path_ends_with_space = (chars_added_by_token[-1] == ' ')
                else: # chars_added_by_token is empty
                    # This implies current_token was a separator and current_path_ends_with_space was already True.
                    # So, current_path_ends_with_space remains True.
                    if not current_token.isalnum(): # This must be a separator token
                        current_path_ends_with_space = True
                    # (An alphanumeric token should always add characters or cause failure)
                
                # Track the first word token in the current span tokens[i...j]
                if current_token.isalnum():
                    if first_word_token_idx_in_span == -1:
                        first_word_token_idx_in_span = j
                
                # If this node in Trie marks an end of a dictionary entry
                if current_node_in_trie.is_end_of_entry:
                    # A match is found. Need to determine the actual annotation range.
                    if first_word_token_idx_in_span == -1:
                        # This should not happen if is_end_of_entry is true, because
                        # normalized Trie keys are non-empty and don't start/end with spaces.
                        # A match implies at least one word token was processed.
                        continue 

                    # Find the actual last word token in the span tokens[first_word_token_idx_in_span...j]
                    actual_last_word_idx_in_match = -1
                    for k_rev in range(j, first_word_token_idx_in_span - 1, -1):
                        if tokens_list[k_rev].isalnum():
                            actual_last_word_idx_in_match = k_rev
                            break
                    
                    # If a valid word-based span is identified for annotation
                    if actual_last_word_idx_in_match != -1:
                        for dict_val in current_node_in_trie.values:
                            # Annotate all tokens from the first word to the last word (inclusive)
                            for k_annotate in range(first_word_token_idx_in_span, actual_last_word_idx_in_match + 1):
                                annotated_results[k_annotate][1].add(dict_val)
            else:
                # Cannot extend path with current_token. Break from inner loop (j).
                break 
    
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

