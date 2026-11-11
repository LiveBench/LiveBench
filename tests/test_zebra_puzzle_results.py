import unittest
from pathlib import Path

from livebench.process_results.reasoning.zebra_puzzle import utils


class ZebraPuzzleResultsTest(unittest.TestCase):
    def test_imports_checkout_source(self):
        self.assertEqual(
            Path(utils.__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "livebench/process_results/reasoning/zebra_puzzle/utils.py",
        )

    def test_rejects_cells_listing_every_candidate(self):
        self.assertEqual(
            utils.zebra_puzzle_process_results(
                "red,1,cat",
                "<solution>red green blue white, 1 2 3 4 5, cat dog bird fish</solution>",
            ),
            0,
        )

    def test_requires_whole_cell_match(self):
        for ground_truth, answer in [
            ("1", "11"),
            ("1", "21"),
            ("red", "red or blue"),
            ("red", "not red"),
            ("cat", "catfish"),
            ("cat", "bobcat"),
            ("green", "greenish"),
        ]:
            with self.subTest(ground_truth=ground_truth, answer=answer):
                self.assertEqual(
                    utils.zebra_puzzle_process_results(
                        ground_truth, f"<solution>{answer}</solution>"
                    ),
                    0,
                )

    def test_preserves_normalized_correct_answers(self):
        for answer in [
            "red,1,cat",
            " RED , 1 , CAT ",
            "red,position1,cat",
            "red,position 1,cat",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(
                    utils.zebra_puzzle_process_results(
                        "red,1,cat", f"<solution>{answer}</solution>"
                    ),
                    1,
                )
        self.assertEqual(
            utils.zebra_puzzle_process_results(
                "light-blue", "<solution>light blue</solution>"
            ),
            1,
        )

    def test_preserves_partial_credit(self):
        self.assertAlmostEqual(
            utils.zebra_puzzle_process_results(
                "red,1,cat", "<solution>red,2,cat</solution>"
            ),
            1 / 3,
        )
        self.assertAlmostEqual(
            utils.zebra_puzzle_process_results(
                "red,1,cat", "<solution>red or blue,1,cat</solution>"
            ),
            1 / 3,
        )
        self.assertAlmostEqual(
            utils.zebra_puzzle_process_results(
                "red,1,cat", "<solution>red,1</solution>"
            ),
            1 / 3,
        )

    def test_preserves_existing_extraction_paths(self):
        for answer in [
            "</solution>red,1,cat</solution>",
            r"\boxed{red,1,cat}",
            "Reasoning.\nred,1,cat",
            "<solution>blue,2,dog</solution><solution>red,1,cat</solution>",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(
                    utils.zebra_puzzle_process_results("red,1,cat", answer), 1
                )
        self.assertEqual(utils.zebra_puzzle_process_results("red,1,cat", ""), 0)

    def test_keeps_legacy_release_evaluator(self):
        self.assertIs(
            utils.get_zebra_puzzle_evaluator("2024-11-24"),
            utils.zebra_puzzle_process_results_old,
        )
        self.assertIs(
            utils.get_zebra_puzzle_evaluator("2024-11-25"),
            utils.zebra_puzzle_process_results,
        )
        self.assertEqual(utils.zebra_puzzle_process_results_old("one", "***1***"), 1)


if __name__ == "__main__":
    unittest.main()
