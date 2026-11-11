import unittest
from pathlib import Path

from livebench.process_results.reasoning.theory_of_mind import utils


class TheoryOfMindResultsTest(unittest.TestCase):
    def test_imports_checkout_source(self):
        self.assertEqual(
            Path(utils.__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "livebench/process_results/reasoning/theory_of_mind/utils.py",
        )

    def test_rejects_hedged_and_negated_answers(self):
        for ground_truth, answer in [
            ("blue_container", "red_container or blue_container"),
            ("red_container", "red_container or blue_container"),
            ("blue_container", "not the blue_container but the red_container"),
            ("blue_container", "not blue_container"),
            ("blue_container", "blue_container red_container green_container"),
            ("blue_container", "dark_blue_container"),
            ("blue_container", "blue_containers"),
        ]:
            with self.subTest(ground_truth=ground_truth, answer=answer):
                self.assertEqual(
                    utils.theory_of_mind_process_results(
                        ground_truth, f"<solution>{answer}</solution>"
                    ),
                    0,
                )

    def test_preserves_case_whitespace_and_punctuation(self):
        for answer in [
            "<solution>blue_container</solution>",
            "<SOLUTION> BLUE_CONTAINER. </SOLUTION>",
            "<solution>\nblue_container!\n</solution>",
            "<solution>blue_container,!?;:</solution>",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(
                    utils.theory_of_mind_process_results(" blue_container ", answer), 1
                )

    def test_preserves_fallback_prefixes(self):
        for prefix in ["", "Answer: ", "answer: ", "The answer is ", "the answer is "]:
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    utils.theory_of_mind_process_results(
                        "blue_container", "Reasoning.\n" + prefix + "blue_container."
                    ),
                    1,
                )

    def test_uses_last_answer_and_malformed_tag_fallback(self):
        self.assertEqual(
            utils.theory_of_mind_process_results(
                "blue_container",
                "<solution>red_container</solution><solution>blue_container</solution>",
            ),
            1,
        )
        self.assertEqual(
            utils.theory_of_mind_process_results(
                "blue_container",
                "<solution>blue_container</solution><solution>red_container</solution>",
            ),
            0,
        )
        self.assertEqual(
            utils.theory_of_mind_process_results(
                "blue_container", "</solution>blue_container</solution>"
            ),
            1,
        )

    def test_wrong_and_empty_answers(self):
        for answer in [
            "",
            "<solution></solution>",
            "<solution>red_container</solution>",
            "Answer: red_container",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(
                    utils.theory_of_mind_process_results("blue_container", answer), 0
                )


if __name__ == "__main__":
    unittest.main()
