import pytest

from livebench.process_results.writing.connections.utils import (
    connections_process_results,
)

GROUPS = [
    ["apple", "pear", "plum", "fig"],
    ["red", "blue", "green", "pink"],
    ["cat", "dog", "cow", "hen"],
    ["one", "two", "six", "ten"],
]
GROUND_TRUTH = ",".join(word for group in GROUPS for word in group)


def score(*groups):
    words = [word for group in groups for word in group]
    return connections_process_results(GROUND_TRUTH, "<solution>" + ", ".join(words) + "</solution>")


@pytest.mark.parametrize(
    "answer_groups,expected",
    [
        ([GROUPS[0], GROUPS[1], GROUPS[2], GROUPS[3]], 1.0),
        ([GROUPS[3], GROUPS[2], GROUPS[1], GROUPS[0]], 1.0),
        ([GROUPS[0], GROUPS[1], ["cat", "dog", "one", "two"], ["cow", "hen", "six", "ten"]], 0.5),
        ([["zero", "zeta", "zed", "zip"]] * 4, 0.0),
    ],
)
def test_distinct_groups_are_scored_as_before(answer_groups, expected):
    assert score(*answer_groups) == expected


@pytest.mark.parametrize(
    "answer_groups,expected",
    [
        ([GROUPS[0]] * 4, 0.25),
        ([GROUPS[0], GROUPS[0], GROUPS[1], GROUPS[1]], 0.5),
        ([GROUPS[0], GROUPS[1], GROUPS[2], GROUPS[3], GROUPS[3]], 1.0),
        ([GROUPS[0], GROUPS[1], GROUPS[2], GROUPS[3]] * 2, 1.0),
    ],
)
def test_a_repeated_correct_group_is_counted_once(answer_groups, expected):
    assert score(*answer_groups) == expected
