# challenge='dictionary-nohint', generator='gpt-4.1-2025-04-14', temperature=0.7
# --- BEGIN GENERATED CODE ---

import collections.abc

# Helper: check if a token is a word token (for matching in phrases)
def is_word_token(token):
    return token.isalnum()

# Helper: normalize tokens for case-insensitive lookup
def norm(token):
    return token.lower()

class TrieNode:
    __slots__ = ("children", "entry")

    def __init__(self):
        self.children = dict()
        self.entry = None  # None or entry/entries

class DictionaryIndex:
    def __init__(self):
        self.root = TrieNode()
        self.single_token = dict()  # normed token -> set of entry ids

    def insert_phrase(self, phrase_tokens, entry):
        node = self.root
        for t in phrase_tokens:
            t_norm = norm(t)
            if t_norm not in node.children:
                node.children[t_norm] = TrieNode()
            node = node.children[t_norm]
        if node.entry is None:
            node.entry = set()
        node.entry.add(entry)
    
    def add_single(self, token, entry):
        t_norm = norm(token)
        if t_norm not in self.single_token:
            self.single_token[t_norm] = set()
        self.single_token[t_norm].add(entry)

def _split_phrase_to_tokens(phrase):
    # Split a phrase string into tokens by separating words and separators
    # E.g., "AAA, BBB" -> ["AAA", ",", "BBB"]
    # We'll treat runs of alnum as a word, everything else as separate separators
    import re
    # This regex splits at word boundaries, keeping separators as tokens.
    # It matches runs of alnum, or single non-alnum chars (including space).
    return [t for t in re.findall(r'\w+|[^\w\s]|\s', phrase) if t]

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    idx = DictionaryIndex()
    for key, entry in dictionary.items():
        key_norm = key.lower()
        # For single-token entries (no separator), treat as word
        if all(c.isalnum() or c.isspace() for c in key):
            # But still split, in case it's a phrase ("AAA BBB")
            tokens = _split_phrase_to_tokens(key)
        else:
            tokens = _split_phrase_to_tokens(key)
        # Remove leading/trailing separators for phrase splitting
        # (but for a single word, this is a no-op)
        # E.g. ", AAA BBB ," -> [",", "AAA", "BBB", ","]
        left = 0
        right = len(tokens)
        while left < right and not is_word_token(tokens[left]):
            left += 1
        while right > left and not is_word_token(tokens[right-1]):
            right -= 1
        core_tokens = tokens[left:right]
        # Decide if this is a phrase or a single word
        if len(core_tokens) == 1:
            idx.add_single(core_tokens[0], entry)
        if core_tokens:
            idx.insert_phrase(core_tokens, entry)
    return idx

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    # Convert tokens to list for indexing
    tokens = list(tokens)
    n = len(tokens)
    annotations = [set() for _ in range(n)]
    idx = dictionary_index

    # 1. Single-token annotations
    for i, token in enumerate(tokens):
        t_norm = norm(token)
        if t_norm in idx.single_token:
            annotations[i].update(idx.single_token[t_norm])

    # 2. Multi-token phrase annotations
    # For each position, try to match any phrase starting there
    for start in range(n):
        node = idx.root
        positions = []  # positions of the matched tokens (including separators)
        has_word = False
        # We'll scan forward, skipping leading separators (they don't start a phrase)
        i = start
        while i < n and not is_word_token(tokens[i]):
            i += 1
        if i >= n:
            continue
        j = i
        node = idx.root
        positions = []
        # Now scan forward, matching tokens along the trie
        last_word_idx = -1
        token_positions = []
        while j < n:
            t_norm = norm(tokens[j])
            if t_norm in node.children:
                node = node.children[t_norm]
                positions.append(j)
                if is_word_token(tokens[j]):
                    last_word_idx = len(positions) - 1  # index in positions
                token_positions.append(tokens[j])
                # If this node is terminal (end of a phrase), annotate its span
                if node.entry is not None:
                    # Determine the span: find first and last word token in positions
                    # Only annotate inner separators, i.e., those between first and last word
                    # The phrase span is positions[first_word_idx : last_word_idx+1]
                    # But as we skipped leading separators, first token is word
                    # Find last word in positions
                    first_word_idx = 0  # always 0
                    lw_idx = last_word_idx
                    # The phrase tokens are positions[first_word_idx:lw_idx+1]
                    phrase_pos = positions[first_word_idx:lw_idx+1]
                    if not phrase_pos:  # safety
                        pass
                    else:
                        # Annotate:
                        # - all word tokens in the phrase
                        # - inner separators (not at start or end)
                        for k, pos in enumerate(positions[first_word_idx:lw_idx+1]):
                            tok = tokens[pos]
                            if is_word_token(tok):
                                annotations[pos].update(node.entry)
                            else:
                                # Only annotate if not at the start or end of phrase
                                if k != 0 and k != lw_idx-first_word_idx:
                                    annotations[pos].update(node.entry)
                        # We continue, since longer matches are possible
                j += 1
            else:
                # Allow separators in between (skip, but record for annotation)
                if not is_word_token(tokens[j]):
                    positions.append(j)
                    token_positions.append(tokens[j])
                    j += 1
                    continue
                else:
                    break

    # Return annotated tokens
    return [(tok, annotations[i]) for i, tok in enumerate(tokens)]


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

