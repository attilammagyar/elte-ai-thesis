# challenge='dictionary-hint', generator='gpt-4.1-2025-04-14', temperature=0.0
# --- BEGIN GENERATED CODE ---

import collections.abc
import re

# Helper: what is a word separator? We'll treat whitespace and most punctuation as separators.
def is_separator(token):
    # A separator is a token that is not alphanumeric (e.g., space, punctuation)
    # For this problem, space and punctuation are separators.
    # We'll treat any token that is not a word (contains at least one alnum) as a separator.
    return not any(c.isalnum() for c in token)

def normalize_token(token):
    # Lowercase for case-insensitive matching
    return token.lower()

def normalize_separator(token):
    # All separators are normalized to a single space
    return ' '

def split_phrase_to_tokens(phrase):
    # Split a phrase into tokens and separators, preserving separators as tokens
    # E.g., "AAA, BBB" -> ["AAA", ",", " ", "BBB"]
    # We'll use a regex to split on word boundaries and keep separators
    # But for this problem, we want to split on spaces and punctuation, keeping them
    # We'll use re.findall to get all word or non-word tokens
    # \w+ matches word, \W matches non-word (including space, punctuation)
    # But we want to split into tokens and separators, so:
    #   - \w+ for words
    #   - [^\w\s] for punctuation
    #   - \s+ for whitespace
    tokens = []
    for m in re.finditer(r'\w+|[^\w\s]|\s+', phrase, re.UNICODE):
        tokens.append(m.group(0))
    return tokens

class TrieNode:
    __slots__ = ['children', 'entries']
    def __init__(self):
        self.children = dict()  # normalized token -> TrieNode
        self.entries = set()    # set of dictionary entry ids

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    # Build a trie for fast phrase lookup
    root = TrieNode()
    for phrase, entry_id in dictionary.items():
        tokens = split_phrase_to_tokens(phrase)
        node = root
        for token in tokens:
            if is_separator(token):
                key = normalize_separator(token)
            else:
                key = normalize_token(token)
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.entries.add(entry_id)
    # For single-token entries, build a lookup for fast individual word lookup
    single_token_map = dict()
    for phrase, entry_id in dictionary.items():
        tokens = split_phrase_to_tokens(phrase)
        # Only if the phrase is a single word (no separators)
        if len(tokens) == 1 and not is_separator(tokens[0]):
            key = normalize_token(tokens[0])
            if key not in single_token_map:
                single_token_map[key] = set()
            single_token_map[key].add(entry_id)
    return {
        'trie': root,
        'single_token_map': single_token_map
    }

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    tokens = list(tokens)
    n = len(tokens)
    annotated = [set() for _ in range(n)]

    trie = dictionary_index['trie']
    single_token_map = dictionary_index['single_token_map']

    # For each position, we keep a list of active candidates:
    # Each candidate is a tuple:
    #   (trie_node, start_index, num_tokens, num_leading_seps, num_trailing_seps, last_token_was_sep)
    # - trie_node: current node in trie
    # - start_index: index in tokens where this candidate started
    # - num_tokens: number of tokens in this candidate so far
    # - num_leading_seps: number of leading separators at the start of the candidate
    # - num_trailing_seps: number of trailing separators at the end of the candidate
    # - last_token_was_sep: bool, whether the last token was a separator (for trailing sep counting)
    candidates = []

    for i, token in enumerate(tokens):
        is_sep = is_separator(token)
        norm_token = normalize_separator(token) if is_sep else normalize_token(token)

        # 1. Start a new candidate from this position
        # For a new candidate, if the token is a separator, we don't start from root (phrases can't start with separator)
        if not is_sep:
            node = trie
            if norm_token in node.children:
                next_node = node.children[norm_token]
                # Check if this node is terminal (single-word phrase)
                if next_node.entries:
                    for eid in next_node.entries:
                        annotated[i].add(eid)
                # Candidate: (node, start, num_tokens, num_leading_seps, num_trailing_seps, last_token_was_sep)
                candidates.append((
                    next_node, i, 1, 0, 0, False
                ))

        # 2. Advance existing candidates
        new_candidates = []
        for cand in candidates:
            node, start, num_tokens, num_leading_seps, num_trailing_seps, last_token_was_sep = cand
            advanced = False
            # Try to advance in trie
            if norm_token in node.children:
                next_node = node.children[norm_token]
                advanced = True
                # If this is a separator, update trailing seps
                if is_sep:
                    if num_tokens == 1 and last_token_was_sep:
                        # Still leading seps
                        new_leading = num_leading_seps + 1
                        new_trailing = 0
                    elif num_tokens == 1:
                        new_leading = 1
                        new_trailing = 0
                    else:
                        new_leading = num_leading_seps
                        new_trailing = num_trailing_seps + 1
                else:
                    new_leading = num_leading_seps
                    new_trailing = 0
                # If this node is terminal, annotate the phrase
                if next_node.entries:
                    # Annotate all tokens in the phrase, except leading/trailing separators
                    phrase_len = num_tokens + 1
                    first = start
                    last = i
                    # Compute indices to annotate: skip leading/trailing seps
                    l = first + new_leading
                    r = last - new_trailing
                    for idx in range(l, r + 1):
                        # For separators, only annotate if not at l or r
                        if is_separator(tokens[idx]):
                            if idx == l or idx == r:
                                continue
                        for eid in next_node.entries:
                            annotated[idx].add(eid)
                # Add advanced candidate
                new_candidates.append((
                    next_node, start, num_tokens + 1, new_leading, new_trailing, is_sep
                ))
            # If can't advance, but token is separator, we may want to extend candidate (for consecutive separators)
            if not advanced and is_sep and norm_token == ' ':
                # Only extend if last token was also a separator or at the start
                # This allows for multiple consecutive separators to be counted as leading/trailing
                if last_token_was_sep or num_tokens == 1:
                    # Don't advance in trie, but increase num_tokens and leading/trailing seps
                    if num_tokens == 1:
                        new_leading = num_leading_seps + 1
                        new_trailing = 0
                    else:
                        new_leading = num_leading_seps
                        new_trailing = num_trailing_seps + 1
                    new_candidates.append((
                        node, start, num_tokens + 1, new_leading, new_trailing, True
                    ))
        candidates = new_candidates

        # 3. Individual token annotation (single-token entries)
        if not is_sep:
            key = norm_token
            if key in single_token_map:
                for eid in single_token_map[key]:
                    annotated[i].add(eid)

    # Output: list of (token, set)
    return [(token, annotated[i]) for i, token in enumerate(tokens)]


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

