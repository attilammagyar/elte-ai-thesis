# challenge='dictionary-nohint', generator='gpt-4.1-2025-04-14', temperature=1.0
# --- BEGIN GENERATED CODE ---

import collections.abc
import re

def build_dictionary_index(dictionary: collections.abc.Mapping):
    """
    Builds a case-insensitive trie index for phrases as well as
    a mapping for single-word/phrase concatenated tokens.
    """
    # Build a trie for phrases (with token/separator distinction)
    # Single token (word/word-piece) mapping: lowercased token -> {dict_entry}
    # For compound word matching: list of normalized (no separators, just tokens) -> {dict_entry}
    trie = dict()
    single_token_map = dict()
    max_phrase_len = 1

    # Helper to determine if a string is a separator token
    def is_separator(token):
        # If a token contains any alphanumeric character, it's not a separator
        return not any(c.isalnum() for c in token)

    # For compound words (e.g. "AAABBB"), store for O(1) lookup
    compound_word_entries = collections.defaultdict(set)
    # For tokenized phrase (with separators), build a trie
    # The trie keys: tuple(token_or_sep, ...), each part being the lowercased original.

    # For splitting dictionary phrases into tokens (preserving separators)
    def split_keep_separators(text):
        # Split on word boundaries, keeping the separators as tokens.
        # This returns a list of alternating word and non-word parts.
        result = []
        # Using regex: \w+ matches words, \W+ matches separators
        for m in re.finditer(r'\w+|\W+', text, re.UNICODE):
            result.append(m.group(0))
        return result

    for phrase, entry_id in dictionary.items():
        phrase_lower = phrase.lower()
        tokens = split_keep_separators(phrase_lower)
        # Remove empty tokens (can happen only if e.g. phrase was all spaces)
        tokens = [t for t in tokens if t]
        if not tokens:
            continue

        if len(tokens) == 1 and not is_separator(tokens[0]):
            # Single token, not a separator -- store for quick lookup
            token = tokens[0]
            if token not in single_token_map:
                single_token_map[token] = set()
            single_token_map[token].add(entry_id)

        # Always enter as a phrase
        node = trie
        token_count = 0
        for tok in tokens:
            t = tok
            if t not in node:
                node[t] = {}
            node = node[t]
            if not is_separator(t):
                token_count += 1
        if "_entries" not in node:
            node["_entries"] = set()
        node["_entries"].add(entry_id)
        max_phrase_len = max(max_phrase_len, len(tokens))

        # For compound words (e.g. "AAABBB"); we only add to the compound-word lookup if the phrase
        # contains only word tokens (no separators) and has more than one token when split as words
        word_tokens = re.findall(r'\w+', phrase_lower)
        if len(word_tokens) > 1:
            compound_word = "".join(word_tokens)
            compound_word_entries[tuple(word_tokens)].add(entry_id)

    # For efficient compound word matching (token concatenation), we store all concat forms.
    # Example: for "AAABBB" entry and tokens ["AAA", "BBB"], if "".join(["AAA", "BBB"]).lower() matches,
    # then we annotate over those tokens.
    compound_concat_map = {}
    for word_tokens, entry_ids in compound_word_entries.items():
        concat = "".join(word_tokens)
        compound_concat_map[concat] = entry_ids.copy()

    index = {
        "trie": trie,
        "single_token_map": single_token_map,
        "max_phrase_len": max_phrase_len,
        "compound_concat_map": compound_concat_map,
        "dictionary": {k.lower(): v for k, v in dictionary.items()},
    }
    return index

def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object):
    trie = dictionary_index["trie"]
    single_token_map = dictionary_index["single_token_map"]
    max_phrase_len = dictionary_index["max_phrase_len"]
    compound_concat_map = dictionary_index["compound_concat_map"]

    tokens = list(tokens)
    n = len(tokens)
    annotations = [set() for _ in range(n)]

    # Precompute for each token whether it's a separator
    def is_separator(token):
        return not any(c.isalnum() for c in token)

    # Build list with lowercased tokens
    lowered_tokens = [t.lower() for t in tokens]

    # 1. Single token matching
    for i, ltok in enumerate(lowered_tokens):
        if ltok in single_token_map and not is_separator(ltok):
            annotations[i].update(single_token_map[ltok])

    # 2. Phrase trie matching (including separators)
    for start in range(n):
        node = trie
        separators_start = is_separator(lowered_tokens[start])
        end = start
        # Limit phrase length
        for offset in range(max_phrase_len):
            idx = start + offset
            if idx >= n:
                break
            t = lowered_tokens[idx]
            if t not in node:
                break
            node = node[t]
            if "_entries" in node:
                # Annotate per phrase matching rules:
                # - All tokens in the matching region get the entry
                # - For separators (is_separator==True between non-separators), only inner separators
                # - Leading/trailing separators: never annotated
                match_entry_ids = node["_entries"]
                end = idx
                # Determine which tokens are annotated for this phrase starting at start
                match_range = range(start, end + 1)
                # Find first and last non-separator in this window
                first_word, last_word = -1, -1
                for j in match_range:
                    if not is_separator(lowered_tokens[j]):
                        if first_word == -1:
                            first_word = j
                        last_word = j
                # All tokens in match_range except leading/trailing separators are annotated
                for j in match_range:
                    if not is_separator(lowered_tokens[j]):
                        annotations[j].update(match_entry_ids)
                    else:
                        # Only annotate if it's between first/last word (not leading/trailing)
                        if first_word < j < last_word:
                            annotations[j].update(match_entry_ids)
            # else, keep traversing further

    # 3. Compound word matching: try to match merges of consecutive tokens (skip separators)
    # E.g. for dict entry "AAABBB", tokens ["AAA", "BBB"] => check "AAABBB"
    # Only match if all tokens are not separators
    # Sliding window up to reasonable phrase length (to catch realistic merged tokens)
    MAX_COMPOUND_WORD_LEN = 8  # hard upper bound to avoid pathological scan

    for start in range(n):
        # Only try if the starting token isn't a separator
        if is_separator(lowered_tokens[start]):
            continue
        compound = ""
        word_token_indices = []
        for offset in range(MAX_COMPOUND_WORD_LEN):
            idx = start + offset
            if idx >= n:
                break
            # Ignore separators
            if is_separator(lowered_tokens[idx]):
                break
            compound += lowered_tokens[idx]
            word_token_indices.append(idx)
            if compound in compound_concat_map:
                entry_ids = compound_concat_map[compound]
                for j in word_token_indices:
                    annotations[j].update(entry_ids)
        # No need to continue if only one token (already handled)

    # Return result
    # For each token, annotate with the set of matched entries (or empty set if no match)
    result = [(tokens[i], annotations[i]) for i in range(n)]
    return result


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

