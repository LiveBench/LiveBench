#!/usr/bin/env python3
"""
Inspect simplified agentic trajectories for a specific model and question ID.

New format expectations (example fields):
- answer.trajectory: JSON string with shape { "info": { ... }, "messages": [ { "role": str, "content": str, ... } ] }

Usage:
python inspect_agentic_traj.py --model <model_name> [<model_name2>] --question-id <question_id> [--max-content-length <length>]

Options:
- --model: Name(s) of the model(s) to inspect (required, 1-2 models)
- --question-id: ID of the question to inspect (required)
- --max-content-length: Max characters to show per message content (default: 4000)
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.table import Table
from rich import box
from rich.columns import Columns

from livebench.model.api_model_config import get_model_config


def load_model_answers(model_name: str) -> list[dict[str, Any]]:
    """
    Load model answers from the specified JSONL file.
    
    Args:
        model_name: Name of the model whose answers to load
        
    Returns:
        List of answer objects from the JSONL file
    """
    model_answer_path_glob = f"data/live_bench/agentic_coding/**/{model_name}.jsonl"

    file_paths = glob.glob(model_answer_path_glob, recursive=True)
    
    if file_paths == []:
        print(f"Error: Could not find answer file for model {model_name}.")
        sys.exit(1)
    
    answers: list[dict[str, Any]] = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    obj: dict[str, Any] = json.loads(line.strip())
                    # Attach source path for later discovery of question.jsonl
                    obj["__answer_path"] = file_path
                    answers.append(obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}. Skipping.")
                    continue
    
    return answers

def find_answer_by_question_id(answers: list[dict[str, Any]], question_id: str) -> dict[str, Any] | None:
    """
    Find the answer object with the specified question ID.
    
    Args:
        answers: List of answer objects
        question_id: ID of the question to find
        
    Returns:
        Answer object with the specified question ID, or None if not found
    """
    for answer in answers:
        if str(answer.get("question_id", "")) == question_id:
            return answer
    
    return None

def truncate_text(content: str, max_length: int) -> str:
    if not content or len(content) <= max_length:
        return content
    suffix = "\n\n[truncated]"
    keep = max_length - len(suffix)
    if keep <= 0:
        return "[truncated]"
    return content[:keep] + suffix


def parse_new_trajectory(answer: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = answer.get("trajectory")
    if raw is None:
        return {}, []
    if isinstance(raw, str):
        try:
            obj: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return {}, []
    elif isinstance(raw, dict):
        obj = raw  # type: ignore[assignment]
    else:
        return {}, []
    info: dict[str, Any] = obj.get("info", {}) or {}
    messages: list[dict[str, Any]] = obj.get("messages", []) or []
    return info, messages


def load_question_fix_patch(answer: dict[str, Any], question_id: str) -> str | None:
    """
    Given an answer object (with attached "__answer_path"), locate the corresponding
    question.jsonl and return the fix_patch field for the matching question_id.
    """
    answer_path_value = answer.get("__answer_path")
    if not isinstance(answer_path_value, str):
        return None
    answer_path = Path(answer_path_value)
    question_file = answer_path.parent.parent / "question.jsonl"
    if not question_file.exists():
        alt = answer_path.parent / "question.jsonl"
        question_file = alt if alt.exists() else question_file
        if not question_file.exists():
            return None
    try:
        with open(question_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if str(obj.get("question_id", "")) == question_id:
                    value = obj.get("fix_patch")
                    if isinstance(value, str):
                        return value
                    # Return JSON-serialized form if fix_patch is not a string
                    if value is not None:
                        try:
                            return json.dumps(value, ensure_ascii=False)
                        except Exception:
                            return str(value)
                    return None
    except Exception:
        return None
    return None


def print_info(info: dict[str, Any], answer: dict[str, Any], console: Console) -> None:
    if not info:
        console.print("[bold red]No info available in trajectory[/bold red]")
        return
    # High-level identifiers
    console.print(f"[bold]Run ID:[/bold] {answer.get('run_id', '-')}")
    console.print(f"[bold]Model ID:[/bold] {answer.get('model_id', '-')}")
    console.print(f"[bold]Answer ID:[/bold] {answer.get('answer_id', '-')}")
    console.print(f"[bold]Timestamp:[/bold] {answer.get('tstamp', '-')}")

    # Exit and model stats table
    stats: dict[str, Any] = info.get("model_stats", {}) or {}
    table = Table(show_header=False, expand=False, box=box.SIMPLE)
    table.add_row("Exit Status", str(info.get("exit_status", "-")))
    table.add_row("API Calls", str(stats.get("api_calls", "-")))
    table.add_row("Total Input Tokens", str(stats.get("total_input_tokens", "-")))
    table.add_row("Total Output Tokens", str(stats.get("total_output_tokens", "-")))
    table.add_row("Instance Cost", str(stats.get("instance_cost", "-")))
    table.add_row("Mini Version", str(info.get("mini_version", "-")))
    console.print(Panel(table, title="Run Info", expand=False))


def print_messages(messages: list[dict[str, Any]], console: Console, max_content_length: int, final_fix_patch: str | None = None, final_answer: str | None = None) -> None:
    if not messages:
        console.print("[bold red]No messages in trajectory[/bold red]")
        return
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "-")
        content = msg.get("content")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        if not content.strip():
            content = final_answer if final_answer else ""
        # content = truncate_text(content, max_content_length)
        console.print(f"\n[bold]Step {i}[/bold]  [dim]({role})[/dim]")
        # For the final message, if fix_patch is available, render side-by-side
        if i == len(messages) and final_fix_patch:
            fp = truncate_text(final_fix_patch, max_content_length)
            total_width = console.size.width if console.size else 0
            # Reserve some space for borders and padding between columns
            panel_width = max(30, (total_width - 6) // 2) if total_width else None
            left_panel = Panel(escape(content), title="Content", expand=False, width=panel_width)
            right_panel = Panel(escape(fp), title="fix_patch", expand=False, width=panel_width)
            console.print(Columns([left_panel, right_panel], expand=True, equal=True, padding=(0, 1)))
        else:
            console.print(Panel(escape(content), title="Content", expand=False))

def main():
    parser = argparse.ArgumentParser(description="Inspect simplified agentic trajectories for a specific model and question ID.")
    parser.add_argument("--model", required=True, nargs='+', help="Name(s) of the model(s) to inspect (1-2 models)")
    parser.add_argument("--question-id", required=True, help="ID of the question to inspect")
    parser.add_argument("--max-content-length", type=int, default=8000, help="Maximum characters to display per message content (default: 4000)")
    args = parser.parse_args()

    if len(args.model) > 2:
        print("Error: Maximum of 2 models can be compared at once.")
        sys.exit(1)

    console = Console()

    console.print(f"\n[bold]Question ID:[/bold] {args.question_id}")

    for model in args.model:
        console.print(f"Loading answers for model [bold]{model}[/bold]...")
        model_name = get_model_config(model).display_name
        answers = load_model_answers(model_name)
        answer = find_answer_by_question_id(answers, args.question_id)
        if not answer:
            console.print(f"[bold red]No answer found for question ID {args.question_id}[/bold red]")
            continue

        info, messages = parse_new_trajectory(answer)
        fix_patch = load_question_fix_patch(answer, args.question_id)
        print_info(info, answer, console)
        console.print(f"\n[bold]Messages:[/bold]")
        print_messages(messages, console, args.max_content_length, final_fix_patch=fix_patch, final_answer=answer['choices'][0]['turns'][0])

if __name__ == "__main__":
    main()