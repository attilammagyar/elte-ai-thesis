# challenge='dictionary-hint', generator='gpt-4.1-2025-04-14', temperature=0.01
# --- BEGIN GENERATED CODE ---

import collections.abc
import string

# Helper: set of characters considered as word separators
_WORD_SEPARATORS = set(string.whitespace + string.punctuation)

def _is_separator(token):
    # A token is a separator if all its characters are in _WORD_SEPARATORS
    return all(c in _WORD_SEPARATORS for c in token) and token != ""

def _normalize_token(token):
    # Lowercase for case-insensitive matching
    return token.lower()

def _tokenize_phrase(phrase):
    # Split a phrase into tokens (words and separators), mimicking input tokenization
    # This is a simple split: we split on runs of separators, keeping them as tokens
    tokens = []
    i = 0
    while i < len(phrase):
        if phrase[i] in _WORD_SEPARATORS:
            # Start of a separator run
            j = i
            while j < len(phrase) and phrase[j] in _WORD_SEPARATORS:
                j += 1
            tokens.append(phrase[i:j])
            i = j
        else:
            # Start of a word run
            j = i
            while j < len(phrase) and phrase[j] not in _WORD_SEPARATORS:
                j += 1
            tokens.append(phrase[i:j])
            i = j
    return tokens

class TrieNode:
    __slots__ = ("children", "entries")
    def __init__(self):
        self.children = dict()  # normalized token -> TrieNode
        self.entries = set()    # set of dictionary entry IDs

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    # Build a trie for phrase lookup, and a word->entry mapping for single-token lookup
    trie_root = TrieNode()
    word_map = dict()  # normalized word -> set of entry IDs

    for key, entry_id in dictionary.items():
        tokens = _tokenize_phrase(key)
        norm_tokens = [_normalize_token(tok) for tok in tokens]
        # Insert into trie
        node = trie_root
        for tok in norm_tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.entries.add(entry_id)
        # If this is a single word (no separators), add to word_map
        if len(norm_tokens) == 1 and not _is_separator(norm_tokens[0]):
            word_map.setdefault(norm_tokens[0], set()).add(entry_id)
    return {"trie": trie_root, "word_map": word_map}

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    tokens = list(tokens)
    n = len(tokens)
    annotated = [set() for _ in range(n)]

    trie_root = dictionary_index["trie"]
    word_map = dictionary_index["word_map"]

    # For each token, check if it's a word in the dictionary (single-token match)
    for i, token in enumerate(tokens):
        norm_token = _normalize_token(token)
        if not _is_separator(token):
            entries = word_map.get(norm_token)
            if entries:
                annotated[i].update(entries)

    # For phrase/compound matching, use a list of active candidates
    # Each candidate: (trie_node, start_idx, matched_token_indices)
    # matched_token_indices: list of indices in tokens that are part of the phrase
    active = []

    for idx, token in enumerate(tokens):
        norm_token = _normalize_token(token)
        is_sep = _is_separator(token)

        # Start new candidates from this position (if not a separator)
        if not is_sep:
            if norm_token in trie_root.children:
                active.append({
                    "node": trie_root.children[norm_token],
                    "matched": [idx],
                    "started_with_sep": False,
                })
        else:
            # If the separator is not at the start, we can start a candidate for separator phrases
            if norm_token in trie_root.children:
                active.append({
                    "node": trie_root.children[norm_token],
                    "matched": [idx],
                    "started_with_sep": True,
                })

        # Advance existing candidates
        new_active = []
        for cand in active:
            node = cand["node"]
            matched = cand["matched"]
            started_with_sep = cand["started_with_sep"]

            # Try to advance with current token
            if idx != matched[-1]:  # Only advance if this token is not already included
                if norm_token in node.children:
                    new_matched = matched + [idx]
                    new_node = node.children[norm_token]
                    new_active.append({
                        "node": new_node,
                        "matched": new_matched,
                        "started_with_sep": started_with_sep,
                    })
            else:
                # Already included this token, keep candidate alive
                new_active.append(cand)

        active = new_active

        # For all candidates, if current node has entries, annotate the matched tokens
        for cand in active:
            node = cand["node"]
            matched = cand["matched"]
            started_with_sep = cand["started_with_sep"]
            if node.entries:
                # For phrase annotation, annotate all matched tokens,
                # but for separators, only annotate inner separators (not leading/trailing)
                # Find indices of matched tokens
                m = len(matched)
                for j, idx2 in enumerate(matched):
                    tok2 = tokens[idx2]
                    if _is_separator(tok2):
                        # Only annotate if not leading or trailing separator
                        if j == 0 or j == m - 1:
                            continue
                    annotated[idx2].update(node.entries)

    # Now, for compound phrases with separators, we need to handle the case where
    # a phrase starts with a word, then has separators, then more words, etc.
    # The above logic only advances candidates if the next token matches a child in the trie.
    # To handle consecutive separators, we need to allow candidates to "wait" at a node if the next token is a separator,
    # and only advance if the separator matches a child in the trie.

    # To handle this, we need to reimplement the candidate logic to allow for separator normalization.
    # Let's do a more robust implementation below.

    # --- Robust phrase matching with separator normalization ---

    # We'll process each position as a possible phrase start.
    # For each start position, we walk forward, matching tokens and separators as per the trie.

    trie = trie_root
    for start in range(n):
        node = trie
        matched_indices = []
        first_token = tokens[start]
        norm_first = _normalize_token(first_token)
        if norm_first not in node.children:
            continue
        node = node.children[norm_first]
        matched_indices.append(start)
        # For each step, try to advance as far as possible
        idx = start + 1
        while True:
            # At each step, if node has entries, annotate matched tokens
            if node.entries:
                m = len(matched_indices)
                for j, idx2 in enumerate(matched_indices):
                    tok2 = tokens[idx2]
                    if _is_separator(tok2):
                        # Only annotate if not leading or trailing separator
                        if j == 0 or j == m - 1:
                            continue
                    annotated[idx2].update(node.entries)
            if idx >= n:
                break
            next_token = tokens[idx]
            norm_next = _normalize_token(next_token)
            if norm_next in node.children:
                node = node.children[norm_next]
                matched_indices.append(idx)
                idx += 1
            else:
                break

    # Output as required
    return [(tok, annotated[i]) for i, tok in enumerate(tokens)]


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

