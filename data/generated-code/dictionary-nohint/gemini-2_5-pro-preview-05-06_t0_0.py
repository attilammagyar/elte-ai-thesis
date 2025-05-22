# challenge='dictionary-nohint', generator='gemini-2.5-pro-preview-05-06', temperature=0.0
# --- BEGIN GENERATED CODE ---

import collections.abc
import re

# Helper class for Trie nodes
class TrieNode:
    def __init__(self):
        self.children = {}  # char -> TrieNode or word_component -> TrieNode
        self.values = set() # Set of dictionary entry identifiers

# Helper class for the dictionary index structure
class DictionaryIndex:
    def __init__(self):
        self.word_trie = TrieNode()    # For single words or concatenated compound words (e.g., "apple", "blackswan")
        self.phrase_trie = TrieNode()  # For phrases (e.g., "black swan")

def _is_word_like(token: str) -> bool:
    """Checks if a token is word-like (contains any alphanumeric character)."""
    return any(c.isalnum() for c in token)

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.
    """
    idx = DictionaryIndex()

    for key_string, entry_id in dictionary.items():
        normalized_key_string = key_string.lower()
        
        if not normalized_key_string.strip(): # Skip empty or all-whitespace keys
            continue

        # Keys with spaces are treated as phrases; components are extracted by splitting by non-alphanumerics.
        # Keys without spaces are treated as words/compound words and stored char-by-char in word_trie.
        if ' ' in normalized_key_string:
            # Phrase key: e.g., "black swan", "once in a blue moon", "AAA, BBB"
            # Components are extracted by splitting by non-alphanumeric characters.
            key_components = tuple(filter(None, re.split(r'[^a-z0-9]+', normalized_key_string)))
            
            if not key_components: # e.g. key was " , " or similar, resulting in no components
                continue

            current_node = idx.phrase_trie
            for component in key_components:
                current_node = current_node.children.setdefault(component, TrieNode())
            current_node.values.add(entry_id)
        else:
            # Word key (single word or concatenated compound word): e.g., "apple", "blackswan"
            word = normalized_key_string
            
            current_node = idx.word_trie
            for char in word: # Store char by char
                current_node = current_node.children.setdefault(char, TrieNode())
            current_node.values.add(entry_id)
            
    return idx

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.
    """
    # Type cast for internal use, assuming dictionary_index is the object returned by build_dictionary_index
    idx_object: DictionaryIndex = dictionary_index 
    
    tokens_list = list(tokens) 
    n = len(tokens_list)
    
    if n == 0:
        return []

    annotated_tokens = [(token, set()) for token in tokens_list]

    for i in range(n):  # Start index of the token subsegment
        for j in range(i, n):  # End index of the token subsegment
            current_sub_tokens = tokens_list[i : j+1]

            # 1. Try to match as a concatenated word (e.g., "apple", "blackswan")
            #    Stored in idx_object.word_trie
            concatenated_parts_normalized = []
            is_valid_for_concatenation = True
            for token_in_subsegment in current_sub_tokens:
                if _is_word_like(token_in_subsegment):
                    concatenated_parts_normalized.append(token_in_subsegment.lower())
                else:
                    is_valid_for_concatenation = False
                    break
            
            if is_valid_for_concatenation and concatenated_parts_normalized:
                full_concatenated_str = "".join(concatenated_parts_normalized)
                
                current_node = idx_object.word_trie
                match_found_for_word = True
                for char_val in full_concatenated_str:
                    if char_val not in current_node.children:
                        match_found_for_word = False
                        break
                    current_node = current_node.children[char_val]
                
                if match_found_for_word and current_node.values:
                    for token_original_idx in range(i, j + 1):
                        annotated_tokens[token_original_idx][1].update(current_node.values)

            # 2. Try to match as a phrase (e.g., "black swan")
            #    Stored in idx_object.phrase_trie
            phrase_components_normalized = []
            first_word_token_offset_in_subsegment = -1
            last_word_token_offset_in_subsegment = -1
            
            for k_offset, token_in_subsegment in enumerate(current_sub_tokens):
                if _is_word_like(token_in_subsegment):
                    if first_word_token_offset_in_subsegment == -1:
                        first_word_token_offset_in_subsegment = k_offset
                    last_word_token_offset_in_subsegment = k_offset
                    phrase_components_normalized.append(token_in_subsegment.lower())
            
            if phrase_components_normalized:
                current_node = idx_object.phrase_trie
                match_found_for_phrase = True
                for component in phrase_components_normalized:
                    if component not in current_node.children:
                        match_found_for_phrase = False
                        break
                    current_node = current_node.children[component]
                
                if match_found_for_phrase and current_node.values:
                    start_annotation_idx = i + first_word_token_offset_in_subsegment
                    end_annotation_idx = i + last_word_token_offset_in_subsegment
                    
                    for token_original_idx in range(start_annotation_idx, end_annotation_idx + 1):
                        annotated_tokens[token_original_idx][1].update(current_node.values)
                        
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

