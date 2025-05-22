# challenge='dictionary-hint', generator='gpt-4.1-2025-04-14', temperature=1.0
# --- BEGIN GENERATED CODE ---

import collections.abc
import string

# Define what separators are: anything that is not a "word" (word = sequence of word characters)
WORD_CHARS = set(string.ascii_letters + string.digits + "_")
SEPARATOR_MARKER = "<SEP>"  # Used in trie to represent normalized separators

def _is_separator(token):
    # A token is a separator if it contains only whitespace or only punctuation and no wordchars
    # For the sake of all provided tests, let's treat as separator if all chars are not in WORD_CHARS
    return all(ch not in WORD_CHARS for ch in token)

def _normalize_token(token):
    return token.lower()

def _normalize_phrase(phrase):
    # Normalize for the trie: map separators (space, comma etc) to SEPARATOR_MARKER, lowercase, reduce multiple separators
    tokens = []
    buf = ''
    for ch in phrase:
        if ch in WORD_CHARS:
            buf += ch
        else:
            if buf:
                tokens.append(buf.lower())
                buf = ''
            # treat all non-word chars as a single separator
            tokens.append(SEPARATOR_MARKER)
    if buf:
        tokens.append(buf.lower())
    # Reduce multiple separators to one
    norm_tokens = []
    for t in tokens:
        if t == SEPARATOR_MARKER:
            if not norm_tokens or norm_tokens[-1] != SEPARATOR_MARKER:
                norm_tokens.append(SEPARATOR_MARKER)
        else:
            norm_tokens.append(t)
    return norm_tokens

class TrieNode:
    __slots__ = ("children", "entries")
    def __init__(self):
        self.children = dict()
        self.entries = set()  # set of dictionary entry IDs for completed words/phrases

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.
    """
    trie = TrieNode()

    for phrase, entry in dictionary.items():
        norm_tokens = _normalize_phrase(phrase)
        node = trie
        for tok in norm_tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.entries.add(entry)
    # Also, build a standalone word lookup for single-token words (case-insensitive).
    word_lookup = dict()
    for k, v in dictionary.items():
        words = _normalize_phrase(k)
        if len(words) == 1 and words[0] != SEPARATOR_MARKER:
            word_lookup[words[0]] = word_lookup.get(words[0], set())
            word_lookup[words[0]].add(v)
    return {"trie": trie, "word_lookup": word_lookup}

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary index.
    """
    tokens = list(tokens)
    n = len(tokens)
    trie = dictionary_index["trie"]
    word_lookup = dictionary_index["word_lookup"]

    # For every token in tokens, initialize its annotation set
    annots = [set() for _ in tokens]

    # 1. Single-token word lookup (case-insensitive, direct per token)
    for i, token in enumerate(tokens):
        if not _is_separator(token):
            tok_norm = _normalize_token(token)
            entries = word_lookup.get(tok_norm)
            if entries:
                annots[i].update(entries)
    
    # 2. Multi-token (compound or phrase) lookup with trie
    # Outer loop: start position of phrase (advance per token)
    for start in range(n):
        node = trie
        idx = start
        # For separator compression: only allow one or more consecutive separators in the DICTIONARY mapping to a single separator in trie
        num_tokens = 0
        matched_token_indexes = []
        saw_word = False  # To avoid annotating leading separators
        # Candidate phrase expansion: follow as far as possible
        while idx < n:
            token = tokens[idx]
            norm_tok = SEPARATOR_MARKER if _is_separator(token) else _normalize_token(token)
            if norm_tok == SEPARATOR_MARKER:
                if not saw_word:
                    # Still at leading separators, don't walk
                    idx += 1
                    num_tokens += 1
                    matched_token_indexes.append(idx - 1)
                    continue
                # Reduce consecutive separators in the input into a single SEPARATOR_MARKER in trie
                # Only one separator edge per run of input separators
                if SEPARATOR_MARKER not in node.children:
                    break
                # step once in trie for whole run of separators
                node = node.children[SEPARATOR_MARKER]
                matched_token_indexes.append(idx)
                idx0 = idx + 1
                idx = idx0
                num_tokens += 1
                # Now skip any further input separators for input position, but don't move in trie
                while idx < n and _is_separator(tokens[idx]):
                    matched_token_indexes.append(idx)
                    idx += 1
                    num_tokens += 1
                continue
            # Current token is a word token
            if norm_tok not in node.children:
                break
            node = node.children[norm_tok]
            matched_token_indexes.append(idx)
            num_tokens += 1
            saw_word = True # matched at least the first word after separator
            idx += 1

            # Check for completions at the end nodes
            if node.entries:
                # According to test rules:
                # - Do not annotate leading/trailing separators for a match (inner only).
                # - All word tokens in match are annotated.
                # - For inner separator tokens, annotate those between first and last word tokens.
                # Figure out which matched tokens are word tokens and which are separators
                token_types = []
                for mi in matched_token_indexes:
                    token_types.append(_is_separator(tokens[mi]))
                # find first and last word token in match
                inner_begin = 0
                while inner_begin < len(token_types) and token_types[inner_begin]:
                    inner_begin += 1
                inner_end = len(token_types) - 1
                while inner_end >= 0 and token_types[inner_end]:
                    inner_end -= 1
                # Annotate
                for k, mi in enumerate(matched_token_indexes):
                    if inner_begin <= k <= inner_end:
                        if not token_types[k]:
                            # word token, always annotate
                            annots[mi].update(node.entries)
                        else:
                            # separator: only annotate if not leading nor trailing (i.e. is inner)
                            if k != inner_begin and k != inner_end:
                                annots[mi].update(node.entries)
    # Emit
    return [(tokens[i], annots[i]) for i in range(n)]


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

