"""
Processing results for the integrals_with_game math task.

This task combines integration problems with stateful game theory.
Answers are expected in <solution></solution> tags and can be
integers or fractions (e.g., "3375", "11326/3").
"""

import re
import warnings
from fractions import Fraction

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for evaluating math tasks. "
        "Please install sympy via pip install sympy"
    )


def extract_solution_tag(llm_answer: str) -> str | None:
    """Extract the answer from <solution></solution> tags."""
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer, re.IGNORECASE | re.DOTALL)
    if solution_matches:
        # Return the last match (in case of multiple)
        return solution_matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize a LaTeX/math answer string for consistent comparison.
    
    This function standardizes various LaTeX representations of the same
    mathematical expression by:
    - Fixing double-backslashes (\\\\frac -> \\frac) from some model outputs
    - Converting fraction variants (\\dfrac, \\tfrac) to standard \\frac
    - Removing delimiter sizing commands (\\left, \\right, \\bigl, etc.)
    - Converting multiplication (\\cdot -> *)
    - Removing LaTeX spacing commands (\\, for thin space, \\; for thick space)
    - Stripping all whitespace (spaces, newlines) for exact string matching
    
    Args:
        answer: A LaTeX or plain math string (e.g., "\\frac{1}{2}", "3/4", "42")
        
    Returns:
        Normalized string with consistent formatting for comparison
        
    Examples:
        >>> normalize_answer("\\dfrac{1}{2}")
        '\\frac{1}{2}'
        >>> normalize_answer("\\left( x \\right)")
        '(x)'
    """
    answer = answer.strip()
    # Fix double-backslashes (some models like amazon.nova-pro output \\frac instead of \frac)
    answer = answer.replace("\\\\", "\\")
    # Normalize fraction variants
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")
    # Remove delimiter sizing commands (don't affect mathematical value)
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\bigl", "")
    answer = answer.replace("\\bigr", "")
    answer = answer.replace("\\Bigl", "")
    answer = answer.replace("\\Bigr", "")
    # Convert multiplication notation
    answer = answer.replace("\\cdot", "*")
    # Remove spacing commands
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    # Remove whitespace
    answer = answer.replace("\n", "")
    answer = answer.replace(" ", "")
    return answer


def parse_answer(answer: str) -> sympy.Expr | None:
    """Parse a math answer string into a sympy expression."""
    answer = normalize_answer(answer)
    
    # Try to parse as a simple fraction first (e.g., "11326/3")
    try:
        if '/' in answer and '\\' not in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = int(parts[0])
                denom = int(parts[1])
                return sympy.Rational(num, denom)
    except (ValueError, TypeError):
        pass
    
    # Try to parse as an integer
    try:
        return sympy.Integer(int(answer))
    except (ValueError, TypeError):
        pass
    
    # Try to parse as a decimal/float
    try:
        return sympy.Rational(answer).limit_denominator(10000000)
    except (ValueError, TypeError, sympy.SympifyError):
        pass
    
    # Try to extract \frac{num}{denom} pattern directly
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer)
    if frac_match:
        try:
            num = int(frac_match.group(1))
            denom = int(frac_match.group(2))
            return sympy.Rational(num, denom)
        except (ValueError, TypeError):
            pass
    
    # Try to parse as latex
    try:
        parsed = parse_latex(answer)
        return parsed
    except Exception:
        pass
    
    # Try parsing with lark backend
    try:
        parsed = parse_latex(answer, backend='lark')
        if hasattr(parsed, 'children'):
            # lark returns a Tree, get the first child
            parsed = parsed.children[0]
        return parsed
    except Exception:
        pass
    
    return None


def is_equiv(x1: sympy.Expr, x2: sympy.Expr) -> bool:
    """Check if two sympy expressions are equivalent."""
    try:
        diff = x1 - x2
        simplified = sympy.simplify(diff)
        if simplified == 0:
            return True
        # Also check numerical equivalence
        if sympy.Abs(simplified).evalf() < 1e-5:
            return True
    except Exception as e:
        warnings.warn(f"Error comparing expressions: {e}")
    return False


def integrals_with_game_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Process results for the integrals_with_game task.
    
    Uses a cascading fallback strategy to extract the model's answer:
    
    1. **<solution> tags** (Primary): Extracts answer from <solution>...</solution> tags.
       Case-insensitive matching handles <Solution>, <SOLUTION>, etc.
       Example: "The answer is <solution>11326/3</solution>"
       
    2. **\\boxed{} format** (Fallback #1): If no solution tags found, looks for 
       LaTeX boxed expressions. Also handles \\fbox{} by converting to \\boxed{}.
       Example: "Therefore, the final answer is $\\boxed{3375}$"
       
    3. **Direct string match** (Fallback #2): Last resort - checks if the normalized
       ground truth appears in the last 200 characters of the response.
       Example: "...so the answer is 3375."
    
    Mathematical equivalence is checked using sympy for the first two methods,
    allowing different representations of the same value to match
    (e.g., \\frac{22652}{6} equals 11326/3).
    
    Args:
        ground_truth: The expected answer (e.g., "3375", "11326/3")
        llm_answer: The model's full response
        debug: Whether to print debug information
        
    Returns:
        1 if correct, 0 if incorrect
    """
    score = 0
    parsed_model_answer = None
    
    # Parse ground truth
    gt_parsed = parse_answer(ground_truth)
    if gt_parsed is None:
        warnings.warn(f"Could not parse ground truth: {ground_truth}")
        return 0
    
    # Extract answer from solution tags
    solution_text = extract_solution_tag(llm_answer)
    
    if solution_text:
        parsed_model_answer = parse_answer(solution_text)
        if parsed_model_answer is not None:
            if is_equiv(gt_parsed, parsed_model_answer):
                score = 1
    
    # If no solution tag or couldn't parse, try other extraction methods
    if score == 0:
        # Try to find answer in boxed format
        from livebench.process_results.util import last_boxed_only_string, remove_boxed
        
        llm_answer_normalized = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer_normalized)
        if last_boxed:
            boxed_content = remove_boxed(last_boxed)
            parsed_model_answer = parse_answer(boxed_content)
            if parsed_model_answer is not None:
                if is_equiv(gt_parsed, parsed_model_answer):
                    score = 1
    
    # Try to find the answer at the end of the response
    if score == 0:
        # Check if ground truth appears directly in the last part of the answer
        last_part = llm_answer[-200:] if len(llm_answer) > 200 else llm_answer
        
        # For simple integer/fraction answers
        gt_normalized = normalize_answer(ground_truth)
        if gt_normalized in normalize_answer(last_part):
            score = 1
    
    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH:", ground_truth)
        if parsed_model_answer is not None:
            print("PARSED MODEL ANSWER:", parsed_model_answer)
        if solution_text:
            print("SOLUTION TAG CONTENT:", solution_text)
        print("END OF OUTPUT:", llm_answer[-300:])
    
    return score

