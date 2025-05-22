# challenge='dictionary-hint', generator='gemini-2.5-pro-preview-05-06', temperature=1.0
# --- BEGIN GENERATED CODE ---

import collections.abc

class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}  # char -> TrieNode
        self.entry_ids: set[int] = set() # Stores dictionary entry IDs for keys ending at this node

def normalize_dict_key(key_str: str) -> str:
    normalized_parts = []
    current_part = []
    
    # Strip leading/trailing whitespace from the original key first.
    # This simplifies logic as we don't need to handle spaces at the very beginning/end of parts list.
    key_str_effective = key_str.strip()
    
    if not key_str_effective: # If key becomes empty after stripping (e.g. "  ", ".")
        return ""

    for char_val in key_str_effective:
        if char_val.isalnum():
            current_part.append(char_val.lower())
        else: # separator char
            if current_part:
                normalized_parts.append("".join(current_part))
                current_part = []
            # Add a single canonical separator ' ' if not already added for this sequence of separators
            if not normalized_parts or normalized_parts[-1] != ' ':
                normalized_parts.append(' ')
    
    if current_part: # Append last word part if any
        normalized_parts.append("".join(current_part))

    # After loop, if normalized_parts ends with a space (e.g. key was "word."), remove it.
    if normalized_parts and normalized_parts[-1] == ' ':
        normalized_parts.pop()
        
    return "".join(normalized_parts)


def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    root = TrieNode()
    for key, entry_id in dictionary.items():
        normalized_key = normalize_dict_key(str(key)) # Ensure key is string
        if not normalized_key: # Skip empty normalized keys
            continue
            
        node = root
        for char_code in normalized_key:
            node = node.children.setdefault(char_code, TrieNode())
        node.entry_ids.add(entry_id) # entry_id is already int from problem context
    return root

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set[int]]]:
    tokens_list = list(tokens)
    n = len(tokens_list)
    if n == 0:
        return []

    annotated_results: list[tuple[str, set[int]]] = [(token, set()) for token in tokens_list]
    root: TrieNode = dictionary_index # Type hint for clarity

    for i in range(n): # i = start_token_index
        current_trie_node = root
        
        # prev_token_in_sequence_was_word_type: tracks the type of tokens_list[k] for k < j in tokens_list[i..j]
        # Initialized for the first token in the sequence (tokens_list[i]).
        # Effectively, it's as if tokens_list[i] is preceded by a separator context.
        prev_token_in_sequence_was_word_type = False

        # Optimization: If the first token of a potential match sequence (tokens_list[i])
        # is a separator, it cannot start a valid match because normalized dictionary keys
        # always start with an alphanumeric character (due to strip and logic of normalize_dict_key).
        if not any(c.isalnum() for c in tokens_list[i]):
            continue

        for j in range(i, n): # j = current_token_index, sequence is tokens_list[i..j]
            token_str = tokens_list[j]
            current_token_is_word_type = any(c.isalnum() for c in token_str)
            
            temp_trie_node_backup = current_trie_node # To restore if current token fails to extend path

            if current_token_is_word_type:
                core_word_part = "".join(c.lower() for c in token_str if c.isalnum())
                
                # This check is crucial: if core_word_part is empty even if token has alnum,
                # it means alnum chars were not actual letters/numbers e.g. some unicode category.
                # Python's .isalnum() handles unicode letters and numbers. So this is mostly defensive.
                if not core_word_part: 
                    break # Cannot process an empty core word part

                # If previous token in sequence tokens_list[i...j-1] was a separator,
                # current_trie_node should already represent a path ending in ' ' (canonical space).
                # If previous token was a word, current_trie_node path ends in an alnum char.
                # We just append characters of core_word_part.
                advancement_possible = True
                for char_code in core_word_part:
                    if char_code in current_trie_node.children:
                        current_trie_node = current_trie_node.children[char_code]
                    else:
                        advancement_possible = False
                        break
                if not advancement_possible:
                    current_trie_node = temp_trie_node_backup # Restore before break
                    break 
            else: # Current token is separator type
                # If the previous token in sequence was a word type (e.g. "word" then "-"),
                # this separator token introduces a canonical space in the trie path.
                if prev_token_in_sequence_was_word_type:
                    if ' ' in current_trie_node.children:
                        current_trie_node = current_trie_node.children[' ']
                    else:
                        current_trie_node = temp_trie_node_backup # Restore
                        break # Path cannot be extended with a space.
                # else (prev token was also sep type): this separator is "absorbed", no change to trie node.
            
            # Update type tracker for the next iteration (when processing tokens_list[j+1])
            prev_token_in_sequence_was_word_type = current_token_is_word_type

            if current_trie_node.entry_ids:
                for entry_id in current_trie_node.entry_ids:
                    for k_idx in range(i, j + 1):
                        k_token_str = tokens_list[k_idx]
                        is_k_token_word_type = any(c.isalnum() for c in k_token_str)
                        
                        if is_k_token_word_type:
                            annotated_results[k_idx][1].add(entry_id)
                        else: # k_token is separator type
                            if k_idx > i and k_idx < j: # Inner separator
                                annotated_results[k_idx][1].add(entry_id)
                                
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

