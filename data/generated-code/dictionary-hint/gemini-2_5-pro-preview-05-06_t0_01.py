# challenge='dictionary-hint', generator='gemini-2.5-pro-preview-05-06', temperature=0.01
# --- BEGIN GENERATED CODE ---

import collections.abc

# Canonical separator used in Trie paths and for normalizing dictionary keys.
CANONICAL_SEPARATOR = ' '

class TrieNode:
    """Represents a node in the Trie for dictionary indexing."""
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        # Stores dictionary entry IDs (meanings) for keys ending at this node.
        self.entries_ending_here: set[object] = set()

def build_dictionary_index(dictionary: collections.abc.Mapping[str, object]) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.

    Parameters:
        dictionary: Mapping strings (keys) to meanings (values).
    """
    trie_root = TrieNode()

    for dict_key_str, entry_id in dictionary.items():
        key_lower = dict_key_str.lower()
        
        path_chars = []
        current_word_chars = []
        
        # Normalize the dictionary key string into a sequence of characters for the Trie path.
        # Alphanumeric parts form words, non-alphanumeric parts are treated as separators.
        # Multiple separators are collapsed into a single CANONICAL_SEPARATOR.
        for char_in_key in key_lower:
            if char_in_key.isalnum():
                current_word_chars.append(char_in_key)
            else: # Separator character found
                if current_word_chars: # Finalize current word part
                    path_chars.extend(current_word_chars)
                    current_word_chars = []
                # Add canonical separator, ensuring no leading/duplicate canonical separators in path_chars
                if path_chars and path_chars[-1] != CANONICAL_SEPARATOR:
                    path_chars.append(CANONICAL_SEPARATOR)
        
        if current_word_chars: # Append any remaining word part
            path_chars.extend(current_word_chars)

        if not path_chars: # Key was empty or contained only separators
            continue

        # Insert the normalized path into the Trie
        node = trie_root
        for char_in_path in path_chars:
            if char_in_path not in node.children:
                node.children[char_in_path] = TrieNode()
            node = node.children[char_in_path]
        node.entries_ending_here.add(entry_id)
        
    return trie_root

def _is_separator_token(token_str: str) -> bool:
    """Checks if a token string consists only of non-alphanumeric characters."""
    return not any(c.isalnum() for c in token_str)

def _advance_node_by_token_chars(
    current_node: TrieNode, 
    normalized_token_str: str, 
    initial_sep_state: bool
) -> tuple[TrieNode | None, bool]:
    """
    Helper to advance a Trie node based on characters of a normalized "word" token.
    Returns (new_node, new_sep_state) or (None, _) if path fails.
    new_sep_state indicates if the path ended with a canonical separator character.
    """
    node = current_node
    sep_state = initial_sep_state # True if last char processed to reach current_node was a separator
    
    for char_code in normalized_token_str:
        if char_code.isalnum():
            if char_code in node.children:
                node = node.children[char_code]
                sep_state = False
            else:
                return None, False # Path failed
        else: # Non-alnum char within a "word" token, e.g. hyphen in "state-of-the-art"
              # Treat it as a canonical separator for Trie traversal.
            if not sep_state: # If not already in separator state from previous char
                if CANONICAL_SEPARATOR in node.children:
                    node = node.children[CANONICAL_SEPARATOR]
                    sep_state = True
                else:
                    return None, False # Path failed: cannot match this internal separator
            # else: already in sep_state, this non-alnum char is consumed without moving in Trie
            
    return node, sep_state

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.

    Parameters:
        dictionary_index:   A dictionary index created by build_dictionary_index()
        tokens:             The tokens to be annotated.

    Return:
        annotated_tokens:   A list containing (token, annotations) pairs for each token in tokens.
    """
    if not isinstance(dictionary_index, TrieNode): # Should be the Trie root
        # Handle cases like empty dictionary leading to a simple TrieNode, or raise error for invalid index.
        # Assuming build_dictionary_index always returns a valid TrieNode.
        pass 
    
    trie_root_node: TrieNode = dictionary_index

    tokens_list = list(tokens)
    n = len(tokens_list)
    annotated_tokens: list[tuple[str, set[object]]] = [(token_str, set()) for token_str in tokens_list]

    if n == 0:
        return []

    # active_matches stores: (trie_node, match_start_token_idx, num_tokens_in_match, last_char_in_trie_path_was_sep)
    active_matches: list[tuple[TrieNode, int, int, bool]] = []

    for token_idx in range(n):
        current_token_str = tokens_list[token_idx]
        normalized_current_token_str = current_token_str.lower()
        current_token_is_separator = _is_separator_token(current_token_str)

        next_active_matches: list[tuple[TrieNode, int, int, bool]] = []

        # 1. Try to extend existing matches
        for prev_trie_node, match_start_idx, num_tokens, prev_path_char_was_sep in active_matches:
            extended_node: TrieNode | None = None
            current_path_char_is_sep: bool = prev_path_char_was_sep

            if current_token_is_separator:
                if not prev_path_char_was_sep: # Last char in Trie path was not a separator
                    if CANONICAL_SEPARATOR in prev_trie_node.children:
                        extended_node = prev_trie_node.children[CANONICAL_SEPARATOR]
                        current_path_char_is_sep = True
                else: # Last char in Trie path was already a separator; consume current separator token
                    extended_node = prev_trie_node # Stay at the same node
                    current_path_char_is_sep = True 
            else: # Current token is a word/word-piece
                result = _advance_node_by_token_chars(prev_trie_node, normalized_current_token_str, prev_path_char_was_sep)
                extended_node, current_path_char_is_sep = result

            if extended_node is not None:
                next_active_matches.append((extended_node, match_start_idx, num_tokens + 1, current_path_char_is_sep))
        
        # 2. Start new matches beginning with the current token
        # (Only if current token is not a separator, as dict keys don't start with CANONICAL_SEPARATOR)
        if not current_token_is_separator:
            # For a new match starting from root, initial_sep_state is True (as if preceded by a separator)
            # This ensures that if a token like "-word" is processed, the leading '-' is handled correctly.
            # However, our _advance_node_by_token_chars expects initial_sep_state to reflect the state *before* this token.
            # Since we start from root, there's no prior char, so True is a safe default.
            result = _advance_node_by_token_chars(trie_root_node, normalized_current_token_str, True)
            new_match_node, new_match_sep_state = result
            if new_match_node is not None:
                next_active_matches.append((new_match_node, token_idx, 1, new_match_sep_state))
        
        active_matches = next_active_matches

        # 3. Process annotations for all current successful matches
        for match_node, match_start_idx, _, _ in active_matches:
            if match_node.entries_ending_here:
                for entry_id in match_node.entries_ending_here:
                    first_word_token_idx_in_match = -1
                    last_word_token_idx_in_match = -1
                    
                    for k in range(match_start_idx, token_idx + 1):
                        if not _is_separator_token(tokens_list[k]):
                            if first_word_token_idx_in_match == -1:
                                first_word_token_idx_in_match = k
                            last_word_token_idx_in_match = k
                    
                    if first_word_token_idx_in_match != -1: # Match contains at least one word token
                        for k_annotate in range(first_word_token_idx_in_match, last_word_token_idx_in_match + 1):
                            annotated_tokens[k_annotate][1].add(entry_id)
                            
    return annotated_tokens


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

