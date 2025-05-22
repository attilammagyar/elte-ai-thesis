# challenge='dictionary-hint', generator='gemini-2.5-pro-preview-05-06', temperature=0.7
# --- BEGIN GENERATED CODE ---

import collections.abc

# Helper class for the Trie structure
class _TrieNode:
    def __init__(self):
        self.children: dict[str, _TrieNode] = {}  # char -> _TrieNode
        self.entries: set[object] = set()  # Set of dictionary entry_ids

# Helper function to identify separator tokens
def _is_separator_token(token_str: str) -> bool:
    if not token_str: # Treat empty string as a separator
        return True
    # A token is a separator if it contains no alphanumeric characters.
    return not any(c.isalnum() for c in token_str)

# Helper function to normalize dictionary keys for Trie insertion
def _normalize_key_for_trie(key: str) -> str:
    key_lower = key.lower()
    normalized_chars = []
    # True if the last character processed was a separator, or if at the start of the key.
    # This helps consolidate multiple separators and trim leading ones.
    last_char_was_separator = True 
    for char_idx in range(len(key_lower)):
        char_val = key_lower[char_idx]
        if char_val.isalnum():
            normalized_chars.append(char_val)
            last_char_was_separator = False
        else: # Character is a separator
            if not last_char_was_separator: # Add only one space for a run of separators
                normalized_chars.append(' ')
            last_char_was_separator = True # Mark that we've just processed a separator
    
    # Remove trailing space if the key ended with separator(s)
    if normalized_chars and normalized_chars[-1] == ' ':
        normalized_chars.pop()
    
    return "".join(normalized_chars)

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.

    Parameters:
        dictionary: Mapping strings (keys) to meanings (values).
    """
    root = _TrieNode()
    for key_string, entry_id in dictionary.items():
        normalized_key = _normalize_key_for_trie(str(key_string)) # Ensure key_string is str
        
        if not normalized_key:
            # Handle cases where original key was empty or normalized to empty.
            # E.g., if key_string itself is "" or consists only of separators like " , ".
            if not key_string: # Original key was empty string
                 root.entries.add(entry_id)
            # Otherwise (non-empty key normalizing to empty, e.g. " , "), skip.
            continue

        current_node = root
        for char_val in normalized_key:
            current_node = current_node.children.setdefault(char_val, _TrieNode())
        current_node.entries.add(entry_id)
    return root

# Helper for annotate function to apply found matches to the annotations list
def _annotate_found_match(
    annotated_tokens_list: list[tuple[str, set]], 
    first_matched_token_idx: int, 
    last_matched_token_idx: int, 
    entries_to_add: collections.abc.Set,
    all_tokens_original_list: list[str] 
    ):
    for k in range(first_matched_token_idx, last_matched_token_idx + 1):
        is_token_k_separator = _is_separator_token(all_tokens_original_list[k])
        
        if is_token_k_separator:
            # Annotate separator only if it's an inner one within the phrase.
            if k > first_matched_token_idx and k < last_matched_token_idx:
                annotated_tokens_list[k][1].update(entries_to_add)
        else: # Word tokens (or word pieces) are always annotated if part of a match
            annotated_tokens_list[k][1].update(entries_to_add)

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.

    Parameters:
        dictionary_index:   A dictionary index created by build_dictionary_index()
        tokens:             The tokens to be annotated.

    Return:
        annotated_tokens:   A list containing (token, annotations) pairs for each token in tokens.
    """
    tokens_list = list(tokens) # Ensure access by index
    n = len(tokens_list)
    if n == 0:
        return [] # Handles empty input token list explicitly
        
    annotated_tokens_result = [(token, set()) for token in tokens_list]
    # Assume dictionary_index is the root _TrieNode
    trie_root: _TrieNode = dictionary_index # type: ignore 

    for i in range(n): # i is the start_token_idx for a potential match span
        current_trie_node: _TrieNode | None = trie_root
        # trie_path_ended_with_char: True if Trie path ends with alnum char, False if space or at root.
        trie_path_ended_with_char = False 

        for j in range(i, n): # j is the end_token_idx for the current span tokens_list[i...j]
            token_str = tokens_list[j]
            
            if _is_separator_token(token_str):
                if not trie_path_ended_with_char:
                    # Multiple separators in input map to one conceptual one if at start or after another sep.
                    # Trie node doesn't change; token j extends the span.
                    pass 
                elif current_trie_node and ' ' in current_trie_node.children:
                    # Trie path ended with a word-char, now sees a separator token. Transition via ' '.
                    current_trie_node = current_trie_node.children[' ']
                    trie_path_ended_with_char = False
                else:
                    # Trie path expected word continuation or end, but got separator where no space transition allowed.
                    current_trie_node = None 
            else: # Token is a word/word-piece
                token_lower = token_str.lower()
                # An empty string token, if not caught by _is_separator_token, would be problematic.
                # Current _is_separator_token("") is True, so empty strings are handled as separators.
                
                temp_node_for_char_sequence = current_trie_node
                current_token_fully_matched_chars = True
                if not temp_node_for_char_sequence : # Should not happen if current_trie_node starts non-None
                    current_token_fully_matched_chars = False

                for char_val in token_lower:
                    if temp_node_for_char_sequence and char_val in temp_node_for_char_sequence.children:
                        temp_node_for_char_sequence = temp_node_for_char_sequence.children[char_val]
                    else:
                        current_token_fully_matched_chars = False
                        break
                
                if current_token_fully_matched_chars:
                    current_trie_node = temp_node_for_char_sequence
                    trie_path_ended_with_char = True # Path now ends with an alnum char
                else:
                    current_trie_node = None # Token's characters could not be fully matched.
            
            if current_trie_node is None:
                # Path broken by current token tokens_list[j]. Stop extending this span.
                break 
            
            if current_trie_node.entries:
                # A valid dictionary key ends at current_trie_node, formed by tokens_list[i...j].
                _annotate_found_match(annotated_tokens_result, i, j, current_trie_node.entries, tokens_list)
                
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

