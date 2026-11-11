"""AIME answers must match a complete nonnegative digit sequence."""

import unittest

from livebench.process_results.math.math_competitions.utils import aime_process_results


class AimeAnswerTests(unittest.TestCase):
    def test_wrong_integer_containing_answer_is_rejected(self):
        for gold, answer in [
            ("017", "The answer is 2017"),
            ("123", "Final answer: 1234"),
            ("123", "Final answer: 5123"),
            ("123", "Final answer: 0123"),
            ("000", "Final answer: 1000"),
            ("123", "Final answer: -123"),
            ("123", "Final answer: \u0661123"),
        ]:
            with self.subTest(gold=gold, answer=answer):
                self.assertEqual(aime_process_results(gold, answer), 0)

    def test_supported_answer_formats_are_unchanged(self):
        for gold, answer in [
            ("017", "017"),
            ("123", "The answer is 123."),
            ("123", "Final answer: +123"),
            ("017", "Final answer: (017)"),
            ("123", "<solution>123</solution>"),
            ("123", "**123**"),
            ("000", "000"),
        ]:
            with self.subTest(gold=gold, answer=answer):
                self.assertEqual(aime_process_results(gold, answer), 1)

    def test_wrong_answer_is_still_incorrect(self):
        for answer in ["Final answer: 042", "<solution>456</solution>", "no answer"]:
            with self.subTest(answer=answer):
                self.assertEqual(aime_process_results("017", answer), 0)

    def test_answer_outside_last_fifty_characters_is_ignored(self):
        self.assertEqual(aime_process_results("123", "123" + " " * 51), 0)

    def test_tail_window_keeps_left_numeric_context(self):
        for prefix in ["5", "-"]:
            answer = prefix + "123" + " " * 47
            self.assertEqual(len(answer), 51)
            with self.subTest(prefix=prefix):
                self.assertEqual(aime_process_results("123", answer), 0)

    def test_answer_at_start_of_tail_window_is_accepted(self):
        answer = "prefix " + "123" + " " * 47
        self.assertEqual(aime_process_results("123", answer), 1)


if __name__ == "__main__":
    unittest.main()
