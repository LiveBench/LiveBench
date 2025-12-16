# adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py

import os
import re
import traceback
from typing import Any, Callable
import warnings
from multiprocessing import Process, Queue

from livebench.process_results.util import last_boxed_only_string, remove_boxed

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.core.relational import Relational
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

try:
    import lark
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`lark` is required for parsing latex. \
please install lark via pip install lark",
    )

def run_with_timeout(func: Callable, args: tuple[Any, ...] = (), timeout: int = 8, debug: bool = False) -> Any:
    def wrapper(queue):
        try:
            if debug:
                print(f"TRYING {func.__name__} with args {' '.join(['<arg>' + str(arg) + '</arg>' for arg in args])}")
            result = func(*args)
            if debug:
                print(f"FINISHED {func.__name__} with result {result}")
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = Queue()
    process = Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Operation timed out")

    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result

def amps_hard_process_results(ground_truth: str, llm_answer: str, debug=False, question: dict | None = None) -> int:
    retval = 0
    parsed_answer = None

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]

    # Strip integration constants (+C) but NOT subscripted constants (C_1, C_2, etc.)
    # Use regex with negative lookahead to avoid stripping C from C_1, C_2, etc.
    # Match +C or + C only when NOT followed by underscore or opening brace
    llm_answer = re.sub(r'\+\s*[Cc](?![_{])', '', llm_answer)
    llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
    llm_answer = llm_answer.replace("\\dfrac", "\\frac")
    llm_answer = llm_answer.replace("\\tfrac", "\\frac")
    # Normalize matrix row separators with spacing (e.g., \\[6pt]) to just \\
    llm_answer = re.sub(r'\\\\\[.*?\]', r'\\\\', llm_answer)
    # Protect matrix row separators before other preprocessing
    llm_answer = llm_answer.replace("\\\\ ", "<!MATRIX_ROW!>")
    llm_answer = llm_answer.replace("\\\\", "<!MATRIX_ROW_NOSPC!>")
    llm_answer = llm_answer.replace("\\left", "")
    llm_answer = llm_answer.replace("\\right", "")
    llm_answer = llm_answer.replace("\\bigl", "")
    llm_answer = llm_answer.replace("\\bigr", "")
    llm_answer = llm_answer.replace("\\Bigl", "")
    llm_answer = llm_answer.replace("\\Bigr", "")
    llm_answer = llm_answer.replace("\\,", "")
    llm_answer = llm_answer.replace("\\;", "")
    llm_answer = llm_answer.replace("\n", "")
    llm_answer = llm_answer.replace("\\cdot", "*")
    # Restore matrix row separators
    llm_answer = llm_answer.replace("<!MATRIX_ROW!>", "\\\\ ")
    llm_answer = llm_answer.replace("<!MATRIX_ROW_NOSPC!>", "\\\\")

    ground_truth = ground_truth.replace("\\left", "")
    ground_truth = ground_truth.replace("\\right", "")
    ground_truth = ground_truth.replace(" ^", "^")
    # Normalize matrix row separators with spacing (e.g., \\[6pt]) to just \\
    ground_truth = re.sub(r'\\\\\[.*?\]', r'\\\\', ground_truth)
    # Don't replace "\\ " when it's a matrix row separator (preceded by \)
    # First, protect matrix row separators by replacing \\ with a placeholder
    ground_truth = ground_truth.replace("\\\\ ", "<!MATRIX_ROW!>")
    ground_truth = ground_truth.replace("\\\\", "<!MATRIX_ROW_NOSPC!>")
    # Now safe to replace backslash-space with multiplication
    ground_truth = ground_truth.replace("\\ ", "*")
    # Restore matrix row separators
    ground_truth = ground_truth.replace("<!MATRIX_ROW!>", "\\\\ ")
    ground_truth = ground_truth.replace("<!MATRIX_ROW_NOSPC!>", "\\\\")

    if '<solution' in llm_answer and '</solution>' in llm_answer:
        solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
        if len(solution_matches) > 0:
            parsed_answer = solution_matches[-1]

    if parsed_answer is None:
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            parsed_answer = normalize_final_answer(remove_boxed(last_boxed))

    if parsed_answer is None:
        # try to extract from the last block of $ $
        last_line = llm_answer.split('\n')[-1]
        if last_line.count('$') >= 2:
            close_pos = last_line.rfind('$')
            if last_line[close_pos - 1] == '$':
                # make sure this works with $$ $$ blocks too
                close_pos -= 1
            open_pos = last_line.rfind('$', 0, close_pos)
            math = last_line[open_pos + 1:close_pos]
            if '=' in math:
                math = math.split('=')[-1].strip()
            elif '\\quad \\text{or} \\quad' in math:
                math = math.split('\\quad \\text{or} \\quad')[-1].strip()
            parsed_answer = normalize_final_answer(math)

    if parsed_answer is not None:
        res = None
        try:
            print('COMPARING', ground_truth, 'WITH', parsed_answer)
            subtask = question.get('subtask') if question else None
            res = is_equiv(ground_truth, parsed_answer, debug=debug, subtask=subtask)
        except TimeoutError:
            warnings.warn("Timeout when comparing ground truth and parsed answer")
        except Exception as e:
            warnings.warn(f"Error when comparing ground truth and parsed answer: {e}")

        # if not res and os.environ.get('OPENAI_API_KEY'):
        #     # Use LLM
        #     res = is_equiv_llm(ground_truth, parsed_answer)

        if res:
            retval = 1
    else:
        if len(llm_answer) > 0 and llm_answer[-1] == '.':
            llm_answer = llm_answer[:-1]
        if ground_truth == llm_answer[-len(ground_truth):]:
            parsed_answer = llm_answer[-len(ground_truth):]
            retval = 1

    if debug and retval == 0:
        print('INCORRECT')
        print('GROUND TRUTH', ground_truth)
        if parsed_answer:
            print('SOLUTION', parsed_answer)
        print('END OF OUTPUT', '\n'.join(llm_answer.split('\n')[-2:]))
    return retval


# class timeout:
#     def __init__(self, seconds=1, error_message="Timeout"):
#         self.seconds = seconds
#         self.error_message = error_message

#     def handle_timeout(self, signum, frame):
#         print('Timeout')
#         raise TimeoutError(self.error_message)

#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)

#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)

def parse(x: str) -> list[sympy.Expr]:
    # Preprocess: Fix common LLM output patterns that confuse the LaTeX parser
    # \cos(x-38)-4\cos(...) gets misparsed because parser thinks -4 is inside first \cos
    # Solution: Add braces to make function boundaries explicit: \cos{(x-38)}
    import re

    # Preprocess: Handle absolute value notation
    # Convert various absolute value notations to \left|...\right| which SymPy understands
    x = x.replace(r'\lvert', r'\left|')
    x = x.replace(r'\rvert', r'\right|')
    x = x.replace(r'\vert', r'|')  # Single \vert -> plain |
    # Note: \mid is typically for "divides" in number theory, not absolute value
    
    # List of trig/math functions that need this fix
    func_names = ['cos', 'sin', 'tan', 'sec', 'csc', 'cot', 
                  'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc', 'arccot',
                  'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'sqrt']
    
    # Build pattern that matches any of these functions
    func_pattern = '|'.join(func_names)
    
    # Find all \func( patterns and wrap them properly with braces
    def wrap_function_args(text):
        result = []
        i = 0
        while i < len(text):
            # Check if we're at a function call
            match = re.match(rf'\\({func_pattern})\(', text[i:])
            if match:
                func_name = match.group(1)
                result.append(f'\\{func_name}{{(')
                i += len(match.group(0))
                
                # Find the matching closing paren
                paren_count = 1
                while i < len(text) and paren_count > 0:
                    if text[i] == '(':
                        paren_count += 1
                    elif text[i] == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            result.append(')}')
                            i += 1
                            continue
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)
    
    original_x = x
    x = wrap_function_args(x)
    
    # If we did preprocessing (x changed), use default backend which handles it better
    # Lark backend creates many ambiguous interpretations and picks the wrong one
    if x != original_x:
        try:
            parsed_xs = parse_latex(x)  # Use default backend
            if isinstance(parsed_xs, lark.Tree):
                parsed_xs = parsed_xs.children
            else:
                parsed_xs = [parsed_xs]
            
            # Apply the pi/E/i fix and return early
            fixed_xs = []
            for expr in parsed_xs:
                expr = expr.subs({
                    sympy.Symbol('pi'): sympy.pi,
                    sympy.Symbol('e'): sympy.E,
                    sympy.Symbol('i'): sympy.I,
                })
                fixed_xs.append(expr)
            return fixed_xs
        except Exception:
            pass  # Fall through to try lark
    
    try:
        # first try to parse normally with lark
        parsed_xs = parse_latex(x, backend='lark')
    except (
        sympy.parsing.latex.errors.LaTeXParsingError,
        sympy.SympifyError,
        TypeError,
        Exception
    ):
        try:
            # this almost only happened for amazon.nova-pro-v1:0 where it outputs e.g. \\frac or \\sqrt all the time
            parsed_xs = parse_latex(x.replace('\\\\', '\\'), backend='lark')
        except Exception as e:
            try:
                # if all else fails, try to parse using default backend
                parsed_xs = parse_latex(x)
            except Exception as e:
                warnings.warn(f"couldn't parse {x} with any backend: {e}")
                return []

    if isinstance(parsed_xs, lark.Tree):
        # lark backend returns multiple options if there is ambiguity
        parsed_xs = parsed_xs.children
    else:
        parsed_xs = [parsed_xs]
    
    # Fix parsing bug: LaTeX parser treats \pi and \e as Symbol('pi') and Symbol('e')
    # instead of the mathematical constants sympy.pi and sympy.E
    # Also, lowercase 'i' in LaTeX should be the imaginary unit I, not a variable
    # Replace these symbols with the actual constants
    fixed_xs = []
    for expr in parsed_xs:
        # Replace Symbol('pi') with sympy.pi, Symbol('e') with sympy.E, Symbol('i') with sympy.I
        expr = expr.subs({
            sympy.Symbol('pi'): sympy.pi,
            sympy.Symbol('e'): sympy.E,
            sympy.Symbol('i'): sympy.I,
        })
        fixed_xs.append(expr)
    
    return fixed_xs


def _compare_tuples(t1: sympy.Tuple, t2: sympy.Tuple, subtask: str | None = None) -> bool:
    if len(t1) != len(t2):
        return False
    
    # First try exact order match (fast path)
    exact_match = True
    for a, b in zip(t1, t2):
        if not is_equiv(sympy.latex(a), sympy.latex(b)):  # type: ignore[arg-type]
            exact_match = False
            break
    if exact_match:
        return True
    
    # Determine if order matters based on subtask
    # Order DOESN'T matter for: polynomial_roots, eigenvalues
    # Order DOES matter for: everything else (solve_linear_system, vertices, etc.)
    order_independent_subtasks = ['polynomial_roots', 'eigenvalues']
    
    if subtask in order_independent_subtasks:
        # Try order-independent comparison
        t2_list = list(t2)
        for elem1 in t1:
            found_match = False
            for i, elem2 in enumerate(t2_list):
                if is_equiv(sympy.latex(elem1), sympy.latex(elem2)):  # type: ignore[arg-type]
                    t2_list.pop(i)  # Remove matched element
                    found_match = True
                    break
            if not found_match:
                return False
        
        # All elements from t1 were matched (and removed from t2_list)
        return len(t2_list) == 0
    
    # For other subtasks, order matters - already checked exact match above
    return False


def _compare_matrices(m1: sympy.MatrixBase, m2: sympy.MatrixBase, subtask: str | None = None) -> bool:
    if m1.shape != m2.shape:
        return False
    
    # First try exact order match (fast path)
    exact_match = True
    for a, b in zip(m1, m2):
        if not is_equiv(sympy.latex(a), sympy.latex(b)):  # type: ignore[arg-type]
            exact_match = False
            break
    if exact_match:
        return True
    
    # Determine if order matters based on subtask
    # Order DOESN'T matter for: polynomial_roots, eigenvalues
    # Order DOES matter for: everything else (solve_linear_system, vertices, etc.)
    order_independent_subtasks = ['polynomial_roots', 'eigenvalues']
    
    # For column vectors (nx1), try order-independent comparison if appropriate
    if m1.shape[1] == 1 and subtask in order_independent_subtasks:
        m2_list = list(m2)
        for elem1 in m1:
            found_match = False
            for i, elem2 in enumerate(m2_list):
                if is_equiv(sympy.latex(elem1), sympy.latex(elem2)):  # type: ignore[arg-type]
                    m2_list.pop(i)
                    found_match = True
                    break
            if not found_match:
                return False
        return len(m2_list) == 0
    
    # For other matrices or when order matters, already checked exact match
    return False


def _rel_to_expr(val):
    if isinstance(val, Relational):
        return val.lhs - val.rhs
    return val


def _get_expected_variables(subtask: str | None = None, expr=None) -> set:
    """
    Get the expected variable symbols for a given subtask.
    Most tasks use 'x', but some use other variables or multiple variables.

    Args:
        subtask: The subtask name (task-level, e.g., 'partial_derivatives')
        expr: Optional expression to inspect for variable detection
    """
    # Tasks that typically use two variables x, y
    # Note: subtask contains the task name, not the subtype
    two_var_tasks = {
        'partial_derivatives',  # PartialDerivatives task
        'implicit_differentiation',  # ImplicitDifferentiation task
    }

    if subtask in two_var_tasks:
        # Check if the expression contains 'z' - if so, it's likely the three_var subtype
        if expr is not None and sympy.Symbol('z') in expr.free_symbols:
            return {sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')}
        # Otherwise, it's a two-variable problem
        return {sympy.Symbol('x'), sympy.Symbol('y')}

    # Derivatives task with parametric subtypes uses 't'
    # Check if 't' is in the expression to detect parametric cases
    if subtask == 'derivatives' and expr is not None:
        if sympy.Symbol('t') in expr.free_symbols:
            return {sympy.Symbol('t')}

    # Default: assume 'x' is the variable
    return {sympy.Symbol('x')}


def _extract_arbitrary_constants(expr, subtask: str | None = None):
    """
    Extract arbitrary constants from an expression (symbols like C_1, C_2, etc.).
    Excludes the independent variable(s) and mathematical constants (e, pi, I, etc.).

    Args:
        expr: The sympy expression
        subtask: The subtask name (used to determine expected variables)
    """
    math_constants = {sympy.E, sympy.pi, sympy.I, sympy.oo, sympy.zoo, sympy.nan}
    var_symbols = _get_expected_variables(subtask, expr=expr)
    free_syms = expr.free_symbols - var_symbols - math_constants
    return sorted(free_syms, key=str)


def _is_equiv_with_constant_permutation(expr1, expr2, debug=False, subtask=None):
    """
    Check if two expressions are equivalent under arbitrary constant renaming.
    This handles cases where models use different constant names (C_1 vs D_1) or
    assign indices in different orders.

    Args:
        expr1: Ground truth expression (sympy.Expr)
        expr2: Model answer expression (sympy.Expr)
        debug: Print debug info
        subtask: The subtask name (used to determine expected variables)

    Returns:
        True if expressions are equivalent under some constant renaming, False otherwise
    """
    import itertools

    try:
        # Extract arbitrary constants from both expressions
        constants1 = _extract_arbitrary_constants(expr1, subtask=subtask)
        constants2 = _extract_arbitrary_constants(expr2, subtask=subtask)

        if debug and (constants1 or constants2):
            print(f"  Ground truth constants: {constants1}")
            print(f"  Answer constants: {constants2}")

        # If different number of constants, they can't be equivalent
        if len(constants1) != len(constants2):
            if debug:
                print(f"  Different number of constants: {len(constants1)} vs {len(constants2)}")
            return False

        # If no constants, fall back to regular comparison
        if len(constants1) == 0:
            return None  # Let other comparison methods handle it

        # Try all permutations of constant mappings
        for perm in itertools.permutations(constants2):
            # Create substitution: map answer constants to ground truth constants
            subs_dict = {ans_const: gt_const for ans_const, gt_const in zip(perm, constants1)}

            # Apply substitution to answer
            # Use two-step substitution with temporary symbols to avoid issues with swaps
            # e.g., {C_1: C_2, C_2: C_1} needs temporaries to work correctly
            temp_subs = {old: sympy.Symbol(f'__TEMP_{i}__') for i, old in enumerate(subs_dict.keys())}
            final_subs = {temp_subs[old]: new for old, new in subs_dict.items()}

            expr2_subst = expr2.subs(temp_subs).subs(final_subs)

            # Check if equivalent
            try:
                diff = sympy.simplify(expr1 - expr2_subst)
                if diff == 0:
                    if debug:
                        print(f"  ✓ Found equivalent mapping: {subs_dict}")
                    return True
            except Exception:
                continue

        if debug:
            print(f"  ✗ No equivalent mapping found")
        return False

    except Exception as e:
        if debug:
            print(f"  Error in constant permutation check: {e}")
        return None  # Let other comparison methods handle it


def _numerical_compare(expr1, expr2, num_samples=50):
    """
    Numerically compare two expressions by evaluating at random points.
    Returns True if they match at all sample points (within epsilon).
    Very fast and works even when symbolic simplification fails.
    
    For indefinite integrals, also checks if expressions differ by only a constant
    (which may be complex), since "up to a constant" is valid.
    """
    import random
    
    try:
        # Get all free symbols (parse() already converts pi, E to constants, so they won't appear here)
        symbols = list(expr1.free_symbols.union(expr2.free_symbols))
        
        if not symbols:
            # No variables, just evaluate directly
            try:
                val1 = complex(expr1.evalf())
                val2 = complex(expr2.evalf())
                return abs(val1 - val2) < 1e-9
            except:
                return None
        
        # Track differences to check if they're constant
        differences = []
        
        # Sample at multiple random points with wider range
        for _ in range(num_samples):
            # Generate random values for each symbol
            subs = {}
            for sym in symbols:
                # Mix of small and large values to catch different behaviors
                if random.random() < 0.7:
                    # Most samples in [-10, 10]
                    subs[sym] = random.uniform(-10, 10)
                else:
                    # Some samples in wider range [-100, 100]
                    subs[sym] = random.uniform(-100, 100)
            
            try:
                # Evaluate both expressions
                val1 = complex(expr1.subs(subs).evalf())
                val2 = complex(expr2.subs(subs).evalf())
                
                # Check if they match (within floating point error)
                diff = val1 - val2
                if abs(diff) > 1e-6 * max(abs(val1), abs(val2), 1):
                    differences.append(diff)
            except:
                # If evaluation fails (e.g., division by zero), try another point
                continue
        
        # If no differences found, they're equal
        if len(differences) == 0:
            return True
        
        # Check if all differences are approximately the same constant
        # This handles cases like log(x+1) vs log(|x+1|) which differ by ±iπ
        if len(differences) > 0:
            first_diff = differences[0]
            constant_diff = all(abs(d - first_diff) < 1e-6 * max(abs(first_diff), 1) for d in differences)
            if constant_diff:
                # They differ by a constant, which is valid for indefinite integrals
                return True
            else:
                print('NUMERICAL COMPARISON FAILED', val1, val2, subs)
                return False
        
        # All sample points matched
        return True
    except Exception:
        return None

def _fast_trig_compare(expr1, expr2):
    """
    Fast comparison for trig expressions by canonicalizing to sin/cos.
    Returns True if equal, False if definitely not equal, None if uncertain.
    """
    try:
        # First, compute the difference
        diff = expr1 - expr2
        
        if diff == 0:
            return True
        
        # For trig expressions, trigsimp is often faster and more effective than rewrite+expand
        # Try trigsimp directly on the difference
        diff_simp = sympy.trigsimp(diff)
        if diff_simp == 0:
            return True
        
        # If trigsimp didn't work, try rewriting to sin/cos canonical form
        try:
            expr1_canon = expr1.rewrite(sympy.sin, sympy.cos)
            expr2_canon = expr2.rewrite(sympy.sin, sympy.cos)
            
            diff_canon = (expr1_canon - expr2_canon)
            
            # Try expand on the canonical difference
            diff_canon_exp = diff_canon.expand()
            if diff_canon_exp == 0:
                return True
            
            # Try trigsimp on canonical form
            diff_canon_simp = sympy.trigsimp(diff_canon_exp)
            if diff_canon_simp == 0:
                return True
        except Exception:
            pass
            
        return None  # Uncertain, need more expensive methods
    except Exception:
        return None

def is_equiv(x1: str, x2: str, debug: bool = False, subtask: str | None = None) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:

        parsed_x1s = parse(x1)
        parsed_x2s = parse(x2)

        if len(parsed_x1s) == 0 or len(parsed_x2s) == 0:
            return False

        errors = []
        for parsed_x1 in parsed_x1s:
            for parsed_x2 in parsed_x2s:

                # Tuple answers: compare elementwise
                if isinstance(parsed_x1, sympy.Tuple) and isinstance(parsed_x2, sympy.Tuple):
                    try:
                        if _compare_tuples(parsed_x1, parsed_x2, subtask=subtask):
                            return True
                        else:
                            continue
                    except Exception as e:
                        errors.append(f"couldn't compare tuples {x1} and {x2}: {e}")
                        continue

                # Matrix answers: compare elementwise
                if isinstance(parsed_x1, sympy.MatrixBase) and isinstance(parsed_x2, sympy.MatrixBase):
                    try:
                        if _compare_matrices(parsed_x1, parsed_x2, subtask=subtask):
                            return True
                        else:
                            continue
                    except Exception as e:
                        errors.append(f"couldn't compare matrices {x1} and {x2}: {e}")
                        continue

                try:
                    left = _rel_to_expr(parsed_x1)
                    right = _rel_to_expr(parsed_x2)
                    diff = left - right

                    # Fast path 1: exact equality
                    if diff == 0:
                        if debug:
                            print('CORRECT BASED ON EXACT EQUALITY:', parsed_x1, parsed_x2)
                        return True
                    # don't continue on failure because this could be nonzero even if the expressions are equal
                except Exception as e:
                    errors.append(f"couldn't subtract {x1} and {x2}: {e}")


                # Fast path 1.5: Check if expressions differ only by arbitrary constant renaming
                # This handles cases like C_1*exp(-3x) vs D_1*exp(-3x) or swapped indices
                # Common for differential equations and integrals
                try:
                    result = _is_equiv_with_constant_permutation(parsed_x1, parsed_x2, debug=debug, subtask=subtask)
                    if result is True:
                        if debug:
                            print('CORRECT BASED ON CONSTANT PERMUTATION')
                        return True
                    elif result is False:
                        # Definitely not equivalent even with constant renaming
                        continue
                    # If result is None, fall through to other methods
                except Exception as e:
                    if debug:
                        print(f"Constant permutation check failed: {e}")
                    pass

                # Fast path 2: numerical comparison (very fast, works for all equivalent expressions)
                # This catches cases where symbolic methods fail (e.g., 1/sec(x) vs cos(x))
                try:
                    result = run_with_timeout(_numerical_compare, args=(parsed_x1, parsed_x2), timeout=30, debug=debug)
                    if result is True:
                        return True
                    else:
                        continue
                except TimeoutError:
                    warnings.warn(f"Numerical comparison timed out for {x1} and {x2}")
                except Exception:
                    pass
                
                # Fast path 3: trig canonicalization (handles trig identities symbolically)
                # This is fast and handles sec, csc, cot, 1/sec=cos, sec^2=1+tan^2, etc.
                try:
                    result = run_with_timeout(_fast_trig_compare, args=(parsed_x1, parsed_x2), timeout=30, debug=debug)
                    if result is True:
                        return True
                    else:
                        continue
                except TimeoutError:
                    warnings.warn(f"Trig comparison timed out for {x1} and {x2}")
                except Exception:
                    pass

                # Slow path: general simplify (with timeout)
                try:
                    def check_simplify_zero():
                        return sympy.simplify(diff) == 0
                    
                    result = run_with_timeout(check_simplify_zero, timeout=30, debug=debug)
                    if result:
                        return True
                    else:
                        continue
                except TimeoutError:
                    warnings.warn(f"Simplify timed out for {x1} and {x2}")
                except Exception as e:
                    errors.append(f"couldn't compare simplified {x1} and {x2}: {e}")

                # Numerical approximation check (very fast, last resort)
                try:
                    def check_simplify_numeric():
                        return sympy.Abs(sympy.simplify(diff)) < 0.001
                    
                    result = run_with_timeout(check_simplify_numeric, timeout=30, debug=debug)
                    if result:
                        return True
                    else:
                        continue
                except TimeoutError:
                    warnings.warn(f"Numeric simplify timed out for {x1} and {x2}")
                except Exception as e:
                    errors.append(f"Had some trouble simplifying when comparing {x1} and {x2}: {e}")
                    
        for error in errors:
            warnings.warn(error)
        return False
    except ImportError as e:
        warnings.warn(e)
        raise
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2}: {e}")
        traceback.print_tb(e.__traceback__)
        return False

def is_equiv_llm(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        import openai
        client = openai.Client()
        response = client.chat.completions.create(model='o3', messages=[
            {
                'role': 'user',
                'content': f'Are these latex expressions equivalent or negatives of each other. Reply yes or no: \n "{x1}" and "{x2}"'
            }
        ])
        if response.choices[0].message.content.lower() == 'yes':
            return True
        return False
    except Exception:
        warnings.warn(f"Failed using LLM to comparing {x1} and {x2}: {e}")
        traceback.print_tb(e.__traceback__)
    return False


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{\[])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
