# challenge='dictionary-nohint', generator='gemini-2.5-pro-preview-05-06', temperature=0.3
# --- BEGIN GENERATED CODE ---

import collections.abc

# Helper function to determine if a token is a separator
def _is_separator(token: str) -> bool:
    """
    A token is considered a separator if it's not purely alphanumeric.
    This heuristic covers spaces, punctuation, etc.
    """
    return not token.isalnum()

def build_dictionary_index(dictionary: collections.abc.Mapping) -> object:
    """
    Build an index from a dictionary for fast lookup of words and
    compound phrases.
    """
    indexed_dict = {} # Stores canonical_key -> set_of_entry_ids

    for key_str, entry_id in dictionary.items():
        word_components = []
        current_component = ""
        # True if the original key string contains any separator character.
        # This helps decide if the key might represent a phrase.
        key_had_any_separator = False
        
        for char_in_key in key_str:
            if char_in_key.isalnum():
                current_component += char_in_key.lower()
            else:
                key_had_any_separator = True
                if current_component:
                    word_components.append(current_component)
                    current_component = ""
        if current_component: # Append last component if any
            word_components.append(current_component)

        if not word_components:
            # Key was empty or consisted only of separators (e.g., ",", "  ").
            # Store its lowercase string form.
            canonical_key = key_str.lower()
        elif key_had_any_separator and len(word_components) > 1:
            # Key is a phrase with multiple word components (e.g., "AAA BBB", "AAA, BBB").
            # Canonical form is a tuple of lowercase word components.
            canonical_key = tuple(word_components)
        else:
            # Key is a single word ("AAA"), a compound word ("AAABBB"),
            # or became a single component after processing (e.g., ",AAA," -> "aaa").
            # Canonical form is a single lowercase string.
            # If word_components is ["aaa", "bbb"] but key_had_any_separator is false,
            # it implies original key was "aaabbb" (no separators), so join them.
            # However, if key was "aaabbb", current_component would be "aaabbb", word_components=["aaabbb"].
            # So, if len(word_components) == 1, it's word_components[0].
            # If len(word_components) > 1 AND not key_had_any_separator, it means the original key
            # was something like "AlphaNumeric" which was split due to case change in a hypothetical
            # more complex tokenizer. Given current alnum split, this case is not hit.
            # Thus, "".join(word_components) is effectively word_components[0] here.
            canonical_key = "".join(word_components)


        if canonical_key not in indexed_dict:
            indexed_dict[canonical_key] = set()
        indexed_dict[canonical_key].add(entry_id)
        
    return indexed_dict


def annotate(tokens: collections.abc.Iterable[str], dictionary_index: object) -> collections.abc.Iterable[tuple[str, collections.abc.Set]]:
    """
    Annotate tokens with entries from the dictionary.
    """
    # Cast dictionary_index to its actual type for internal use.
    idx_dict: dict[str | tuple[str, ...], set] = dictionary_index # type: ignore

    token_list = list(tokens) # Work with a list for indexed access.
    n = len(token_list)
    if n == 0:
        return []

    # Initialize result list with original tokens and empty annotation sets.
    annotated_tokens = [(token, set()) for token in token_list]

    for i in range(n):
        # 1. Match individual token token_list[i]
        # Any token (word or separator) can be an entry if its lowercase form is a string key in the index.
        current_token_lower = token_list[i].lower()
        if current_token_lower in idx_dict:
            entry_ids = idx_dict.get(current_token_lower)
            if entry_ids and isinstance(idx_dict[current_token_lower], set): # Ensure it's a direct match, not a phrase
                 annotated_tokens[i][1].update(entry_ids)

        # 2. Match concatenated tokens (compound words like "AAABBB")
        # These are formed by sequences of non-separator tokens.
        current_concatenation = ""
        concat_token_indices = [] # Indices of tokens forming the current_concatenation
        for j in range(i, n):
            token_j_original = token_list[j]
            if _is_separator(token_j_original):
                break # Separators break compound word formation.

            current_concatenation += token_j_original.lower()
            concat_token_indices.append(j)

            if current_concatenation in idx_dict:
                entry_ids = idx_dict.get(current_concatenation)
                # Ensure it's a string key match (implicitly true by current_concatenation type)
                if entry_ids and isinstance(idx_dict[current_concatenation], set):
                    for k_idx in concat_token_indices: # Annotate all tokens part of this compound word
                        annotated_tokens[k_idx][1].update(entry_ids)
        
        # 3. Match phrases (e.g., "AAA BBB" from tokens ["AAA", " ", "BBB"])
        # A phrase match cannot start with a separator token (handles leading separators).
        if _is_separator(token_list[i]):
            continue

        current_phrase_word_components = [] # Lowercase word parts of the potential phrase
        # Indices of all tokens (words and inner separators) involved in the current phrase window.
        phrase_involved_token_indices = [] 
        
        for j in range(i, n):
            token_j_original = token_list[j]
            token_j_lower = token_j_original.lower()
            is_current_token_separator = _is_separator(token_j_original)

            if not is_current_token_separator:
                # It's a word-like token; add to components and track its index.
                current_phrase_word_components.append(token_j_lower)
                phrase_involved_token_indices.append(j)

                # Check if this sequence of word components forms a known phrase.
                if current_phrase_word_components: # Must have at least one word component.
                    canonical_phrase_key = tuple(current_phrase_word_components)
                    if canonical_phrase_key in idx_dict:
                        entry_ids = idx_dict.get(canonical_phrase_key)
                        if entry_ids: # Should be true if key exists.
                            # Annotate all tokens forming this phrase instance.
                            # `j` is the index of the last word component of this match.
                            # All tokens in `phrase_involved_token_indices` are part of this match
                            # (from token_list[i] up to token_list[j], including inner separators).
                            for k_idx in phrase_involved_token_indices:
                                annotated_tokens[k_idx][1].update(entry_ids)
            else:
                # It's a separator token.
                # If it follows word components, it's an inner or trailing separator.
                if current_phrase_word_components: 
                    phrase_involved_token_indices.append(j) # Track its index.
                    
                    # A phrase match might have just ended before this separator.
                    # Re-check with existing word components.
                    canonical_phrase_key = tuple(current_phrase_word_components)
                    if canonical_phrase_key in idx_dict:
                        entry_ids = idx_dict.get(canonical_phrase_key)
                        if entry_ids:
                            # This separator (at index j) is a TRAILING separator for the phrase
                            # formed by `current_phrase_word_components`.
                            # We must annotate only tokens up to the actual last word of that phrase.
                            actual_last_word_idx_in_span = -1
                            # Find the index of the last word token in the current span.
                            for token_idx_in_span in reversed(phrase_involved_token_indices):
                                if not _is_separator(token_list[token_idx_in_span]):
                                    actual_last_word_idx_in_span = token_idx_in_span
                                    break
                            
                            if actual_last_word_idx_in_span != -1:
                                for k_idx in phrase_involved_token_indices:
                                    if k_idx <= actual_last_word_idx_in_span:
                                        # Annotate only up to (and including) the last word component.
                                        annotated_tokens[k_idx][1].update(entry_ids)
                                    # Else: k_idx is for a token after the last word (a trailing separator), so don't annotate.
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

