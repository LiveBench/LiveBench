"""Offline regressions for the LiveCodeBench standard-I/O grader."""

import signal
import sys
import unittest

from livebench.lcb_runner.evaluation.testing_util import (
    Capturing,
    call_method,
    grade_call_based,
    grade_stdio,
    timeout_handler,
)


@unittest.skipUnless(hasattr(signal, "SIGALRM"), "The grader requires POSIX alarms")
class StdioGradingTests(unittest.TestCase):
    def setUp(self):
        self.previous_handler = signal.signal(signal.SIGALRM, timeout_handler)

    def tearDown(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def assert_grades(self, code, inputs, outputs):
        results, metadata = grade_stdio(code, inputs, outputs, timeout=2)
        self.assertEqual(results, [True] * len(inputs), metadata)

    def test_binary_input_methods(self):
        for expression in (
            "sys.stdin.buffer.read()",
            "sys.stdin.buffer.readline() + sys.stdin.buffer.read()",
            "b''.join(sys.stdin.buffer.readlines())",
            "b''.join(sys.stdin.buffer)",
        ):
            with self.subTest(expression=expression):
                self.assert_grades(
                    f"import sys\nprint(sum(map(int, ({expression}).split())))",
                    ["1 2\n3\n", "4\n5 6"],
                    ["6\n", "15\n"],
                )

    def test_binary_output_preserves_text_write_order(self):
        self.assert_grades(
            "import sys\n"
            "print('one', end=' '); sys.stdout.buffer.write(b'two '); "
            "sys.stdout.write('three\\n')",
            [""],
            ["one two three\n"],
        )

    def test_utf8_binary_round_trip(self):
        self.assert_grades(
            "import sys\nsys.stdout.buffer.write(sys.stdin.buffer.read())",
            ["café\nשלום\n"],
            ["café\nשלום\n"],
        )

    def test_text_reads_consume_one_stream_and_preserve_eof(self):
        self.assert_grades(
            "import sys\n"
            "print(input()); print(repr(sys.stdin.read(1))); "
            "print(repr(sys.stdin.readline())); "
            "print(repr(sys.stdin.readlines())); print(repr(sys.stdin.read()))",
            ["alpha\nbc\nde\n"],
            ["alpha\n'b'\n'c\\n'\n['de\\n']\n''\n"],
        )

    def test_binary_and_text_reads_share_input(self):
        self.assert_grades(
            "import sys\n"
            "print(sys.stdin.buffer.read(1).decode()); print(sys.stdin.read())",
            ["abc"],
            ["a\nbc\n"],
        )

    def test_input_iteration_and_eof(self):
        self.assert_grades(
            "import sys\n"
            "print(''.join(sys.stdin), end=''); print(repr(sys.stdin.readline()))",
            ["a\nb\n"],
            ["a\nb\n''\n"],
        )

    def test_capture_survives_stdout_close(self):
        self.assert_grades(
            "import sys\nprint('done'); sys.stdout.close()", [""], ["done\n"]
        )

    def test_streams_are_restored_after_system_exit(self):
        stdin, stdout = sys.stdin, sys.stdout

        def solution():
            print(input())
            raise SystemExit(0)

        with Capturing() as captured:
            call_method(solution, ["done"])
        self.assertEqual(captured, ["done\n"])
        self.assertIs(sys.stdin, stdin)
        self.assertIs(sys.stdout, stdout)

    def test_stdout_is_restored_after_exception(self):
        stdout = sys.stdout
        with self.assertRaisesRegex(ValueError, "solution failed"):
            with Capturing():
                raise ValueError("solution failed")
        self.assertIs(sys.stdout, stdout)

    def test_wrong_answer_and_runtime_error_remain_failures(self):
        for code, expected in (("print(2)", -2), ("raise ValueError('bad')", -4)):
            with self.subTest(code=code):
                results, metadata = grade_stdio(code, [""], ["1\n"], timeout=2)
                self.assertEqual(results, [expected], metadata)

    def test_timeout_remains_a_failure(self):
        results, metadata = grade_stdio("while True: pass", [""], [""], timeout=1)
        self.assertEqual(results, [-3], metadata)

    def test_call_based_grading_still_works(self):
        results, metadata = grade_call_based(
            "def add(a, b): return a + b", ["1\n2", "3\n4"], ["3", "7"], "add", 2
        )
        self.assertEqual(results, [True, True], metadata)


if __name__ == "__main__":
    unittest.main()
