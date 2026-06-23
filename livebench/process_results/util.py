

from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")

    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1].replace("$", "").replace("fbox","boxed")

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def levenshtein_distance(a, b):
    """Edit distance between two sequences (lists, tuples, or strings).

    Pure-Python so it has NO dependency on the ``Levenshtein``/``python-Levenshtein``
    package, whose accepted input types vary by version: the legacy C extension
    accepted lists of ints, but modern ``Levenshtein`` (>=0.21, rapidfuzz-based)
    accepts only str/bytes. Comparing elements with ``==`` means this works directly
    on the integer token sequences the olympiad scorer builds, on any environment.
    """
    n, m = len(a), len(b)
    # dp[i][j] = edit distance between a[:i] and b[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Base cases: transforming to/from an empty sequence costs one op per element.
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]          # tokens match: no cost
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],     # deletion
                                   dp[i][j - 1],     # insertion
                                   dp[i - 1][j - 1]) # substitution
    return dp[n][m]
