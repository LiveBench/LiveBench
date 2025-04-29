# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import itertools
import multiprocessing
import os
import sys
import tempfile
import time
import types
import unittest
import io
import contextlib
from multiprocessing import Array, Value, Manager
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from livebench.bcb_runner.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    redirect_stdin,
    WriteOnlyStringIO,
    swallow_subprocess_output,
    time_limit,
    safe_environment,
    TIMEOUT_LIMIT,
)


def compatible_eval_result(results: Dict) -> Dict:
    # compatibility
    for task_results in results["eval"].values():
        # update the "files" field to "nfiles"
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results


# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Array
):
    with safe_environment(), create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins
        
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        # Create string IO objects to capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        stdin_capture = WriteOnlyStringIO()

        try:
            full_code = code + "\n" + test_code

            # Use contextlib to redirect stdout and stderr instead of swallowing IO
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture), redirect_stdin(new_target=stdin_capture), swallow_subprocess_output():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)
            
            # Capture stdout and stderr content
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = (test.shortDescription(), trace)

            if issues:
                # Store outputs in details
                details["_captured_stdout_"] = stdout_content
                details["_captured_stderr_"] = stderr_content
            
            stat.value = _SUCCESS
        except BaseException as e:
            # Capture stdout and stderr content before the exception
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            # Store outputs and exception details
            details["_captured_stdout_"] = stdout_content
            details["_captured_stderr_"] = stderr_content
            details["_exception_"] = str(e)
            
            stat.value = _FAILED
            
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    min_time_limit: float = 10,
    gt_time_limit: float = 60
) -> Tuple[str, np.ndarray]:
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = min_time_limit + 1
    # shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    # convert details to a dict
    details = dict(details)
    
    if not stat:
        stat = TIMEOUT
        
    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details


def evaluate_files(
    files: List[str],
    inputs: List,
    entry_point: str,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> List[Tuple[str, List[bool]]]:
    ret = []
    # sort files by the id in name (i.e., "../n.py")
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    for file in files:
        code = open(file, "r").read()
        stat, det = untrusted_check(
            code,
            inputs,
            entry_point,
        )
        ret.append((stat, det.tolist()))
    return ret

def trusted_exec(code, test_code, task_id, max_as_limit, max_data_limit, max_stack_limit, times):
    """Execute trusted code in place."""
    # Specify a unique cache dir by modifying XDG_CONFIG_HOME
    old_xdg = os.environ.get("XDG_CONFIG_HOME")
    temp_xdg = tempfile.mkdtemp(prefix="xdg_config_")
    os.environ["XDG_CONFIG_HOME"] = temp_xdg

    try:
        with create_tempdir():
            import shutil
            import builtins

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir
            module_name = "__test__"
            new_module = types.ModuleType(module_name)

            reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

            # Set necessary attributes for the module
            new_module.__dict__.update({
                '__builtins__': builtins,
                '__file__': f"{module_name}.py",
                '__package__': None,
                '__doc__': None,
                'sys': sys,
                'os': os,
                'environ': os.environ,
            })

            # Combine the user code and the test code
            full_code = code + "\n" + test_code

            # Compile and execute the combined code within the new module
            exec(compile(full_code, f"{module_name}.py", 'exec'),
                 new_module.__dict__)
            sys.modules[module_name] = new_module
            TestCases = getattr(new_module, 'TestCases')
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestCases)
            test_result = unittest.TestResult()
            start = time.time()
            with safe_environment(), swallow_io(), time_limit(seconds=TIMEOUT_LIMIT):
                suite.run(test_result)

            errors = test_result.failures + test_result.errors
            if len(errors) > 0:
                print(errors)
                times.value = -1
            else:
                times.value = time.time() - start

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    finally:
        # Restore the original environment variable
        if old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = old_xdg
        shutil.rmtree(temp_xdg, ignore_errors=True)


def trusted_check_exec(code, inputs):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=TIMEOUT_LIMIT):
            trusted_exec(code, inputs)
    except Exception:
        return False
    return True


def trusted_check(
    code: str,
    test_code: str,
    task_id: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    min_time_limit: float = 10,
):
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), min_time_limit) + 1
    # shared memory objects
    times = Value("d", -1)
    manager = Manager()

    p = multiprocessing.Process(
        target=trusted_exec,
        args=(
            code,
            test_code,
            task_id,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            times,
        ),
    )
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    if times.value == -1:
        times = None
    else:
        times = times.value
    
    return {"task_id": task_id, "time": times}