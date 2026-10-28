import unittest

from livebench.process_results.data_analysis.cta.utils import cta_process_results
from livebench.process_results.util import last_boxed_only_string, remove_boxed


class LastBoxedOnlyStringTest(unittest.TestCase):
    def test_braced_answer_after_space_answer(self):
        cases = [
            (r"Earlier: \boxed 1$. Final: \boxed{2}", r"\boxed{2}"),
            (
                r"Earlier: \boxed 1$. Final: \boxed{\frac{1}{2}}",
                r"\boxed{\frac{1}{2}}",
            ),
            (
                r"First: \boxed 1$. Next: \boxed 2$. Final: \boxed{3}",
                r"\boxed{3}",
            ),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(last_boxed_only_string(response), expected)

    def test_supported_answer_forms(self):
        cases = [
            (r"Final: \boxed 2$", "2"),
            (r"Final: \boxed{2}", "2"),
            (r"Final: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            (r"Earlier: \boxed{1}. Final: \boxed{2}", "2"),
            (r"Earlier: \boxed 1$. Final: \boxed 2$", "2"),
            (r"Earlier: \boxed{1}. Final: \boxed 2$", "2"),
            (r"Final: \fbox{2}", "2"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    remove_boxed(last_boxed_only_string(response)), expected
                )
        self.assertIsNone(last_boxed_only_string("No boxed answer"))

    def test_incomplete_final_answer_does_not_fall_back_to_earlier_answer(self):
        self.assertIsNone(
            last_boxed_only_string(r"Earlier: \boxed 1$. Final: \boxed{2")
        )

    def test_cta_scores_correct_final_answer(self):
        response = r"Initially: \boxed city$. Final: \boxed{country}"
        self.assertEqual(cta_process_results("country", response), 1)

    def test_cta_rejects_incorrect_final_answer(self):
        response = r"Initially: \boxed country$. Final: \boxed{city}"
        self.assertEqual(cta_process_results("country", response), 0)


if __name__ == "__main__":
    unittest.main()
