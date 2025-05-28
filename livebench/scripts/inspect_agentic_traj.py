#!/usr/bin/env python3
"""
Inspect agentic trajectories for a specific model and question ID.

Usage:
python inspect_agentic_traj.py --model <model_name> [<model_name2>] --question-id <question_id> [--generate-feedback] [--feedback-model <model_name>] [--max-tool-response-length <length>]

Options:
  --model: Name(s) of the model(s) to inspect (required, 1-2 models)
  --question-id: ID of the question to inspect (required)
  --generate-feedback: Generate LLM feedback on the agent trajectory (optional)
  --feedback-model: Model to use for generating feedback (default: gpt-4-turbo)
  --max-tool-response-length: Maximum length for tool responses before truncation (default: 1000)
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Literal, Any
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.table import Table
from rich import box
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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

    file_paths = glob.glob(model_answer_path_glob)
    
    if file_paths == []:
        print(f"Error: Could not find answer file for model {model_name}.")
        sys.exit(1)
    
    answers = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    answer_obj = json.loads(line.strip())
                    answers.append(answer_obj)
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

def print_history(history: list[dict[str, Any]], console: Console, include_reasoning: bool = True) -> None:
    """
    Print the history field step by step.
    
    Args:
        history: List of history items
        console: Rich console for output
    """
    if not history:
        console.print("No history found in the answer.", style="bold red")
        return

    if isinstance(history, str):
        history = json.loads(history)
    
    i = 0
    for step in history:
        console.print(f"\n[bold]Step {i+1}:[/bold]")
        
        # Print role
        if "role" in step:
            console.print(f"[bold cyan]Role:[/bold cyan] {step['role']}")

        
        # Print content with proper formatting
        if "content" in step:
            content = escape(str(step["content"]))
            console.print(Panel(content, title="Content", expand=False))
        
        # Print Any other fields that might be present
        for key, value in step.items():
            if key == 'reasoning' and isinstance(value, list):
                for r in value:
                    if isinstance(r, dict) and 'encrypted_content' in r:
                        del r['encrypted_content']
            if key not in ["role", "content"] and (include_reasoning or key != 'reasoning'):
                console.print(f"[bold cyan]{key}:[/bold cyan] {escape(str(value))}")
        
        if step['role'] == 'assistant':
                i += 1

def print_trajectory(trajectory: list[dict[str, Any]], console: Console, include_reasoning: bool = True) -> None:
    """
    Print the trajectory field step by step.
    
    Args:
        trajectory: List of history items
        console: Rich console for output
    """
    if not trajectory:
        console.print("No trajectory found in the answer.", style="bold red")
        return

    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    
    for i, step in enumerate(trajectory):
        console.print(f"\n[bold]Step {i+1}:[/bold]")

        
        response = escape(str(step["response"]))
        console.print(Panel(response, title="Response", expand=False))
        
        # Print Any other fields that might be present
        for key, value in step.items():
            if key == 'reasoning' and isinstance(value, list):
                for r in value:
                    if isinstance(r, dict) and 'encrypted_content' in r:
                        del r['encrypted_content']
            if key != 'response' and (include_reasoning or key != 'reasoning'):
                console.print(f"[bold cyan]{key}:[/bold cyan] {escape(str(value))}")
        

def truncate_tool_response(content: str, max_length: int = 1000) -> str:
    """
    Truncate long tool responses to fit within context limits.
    
    Args:
        content: The content to potentially truncate
        max_length: Maximum allowed length for the content
        
    Returns:
        Truncated content if necessary, or original content if within limits
    """
    if not content or len(content) <= max_length:
        return content
    
    truncation_message = "\n\n[Tool Call response truncated to fit context length. This truncation only happens for feedback generation; during actual inference, the agent received the full response]"
    
    # Calculate how much content we can keep, leaving room for the truncation message
    keep_length = max_length - len(truncation_message)
    if keep_length <= 0:
        return truncation_message.strip()
    
    # Keep the first part of the content and add the truncation message
    return content[:keep_length] + truncation_message

def format_history_for_llm(history: list[dict[str, Any]], max_tool_response_length: int = 1000) -> list[dict[str, Any]]:
    """
    Format the history for LLM consumption, extracting role, content, and tool results.
    Long tool responses are truncated to fit within context limits.
    
    Args:
        history: List of history items
        max_tool_response_length: Maximum length for tool responses before truncation
        
    Returns:
        Formatted history suitable for LLM analysis
    """
    if isinstance(history, str):
        history = json.loads(history)
    
    formatted_history = []
    
    for step in history:
        formatted_step = {}
        
        # Include role and content
        if "role" in step:
            formatted_step["role"] = step["role"]
        
        # For tool responses (role=tool), truncate content if too long
        if "content" in step:
            if step.get("role") == "tool":
                formatted_step["content"] = truncate_tool_response(step["content"], max_tool_response_length)
            else:
                formatted_step["content"] = step["content"]
        
        # Include tool calls and results if present
        if "tool_calls" in step:
            formatted_step["tool_calls"] = step["tool_calls"]
        
        if "tool_call_id" in step:
            formatted_step["tool_call_id"] = step["tool_call_id"]
            
        if "name" in step:
            formatted_step["name"] = step["name"]
            
        if "function" in step:
            formatted_step["function"] = step["function"]
        
        formatted_history.append(formatted_step)
    
    return formatted_history

def format_trajectory_for_llm(trajectory: list[dict[str, Any]], max_tool_response_length: int = 1000) -> list[dict[str, Any]]:
    """
    Format the trajectory for LLM consumption, extracting role, content, and tool results.
    Long tool responses are truncated to fit within context limits.
    
    Args:
        traejctory: List of trajectory items
        max_tool_response_length: Maximum length for tool responses before truncation
        
    Returns:
        Formatted trajectory suitable for LLM analysis
    """
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    
    formatted_trajectory = []
    
    for step in trajectory:
        formatted_step = {}
        
        formatted_step['action'] = step['action']

        formatted_step['thought'] = step['thought']

        formatted_step['observation'] = truncate_tool_response(step['observation'], max_length=max_tool_response_length)
        
        formatted_trajectory.append(formatted_step)
    
    return formatted_trajectory

def generate_llm_feedback(history_or_trajectory: list[dict[str, Any]], model: str = "gpt-4.1", max_tool_response_length: int = 1000, mode: Literal['history', 'trajectory'] = 'history') -> str:
    """
    Generate feedback on the agent's trajectory using an LLM.
    
    Args:
        history: The agent's interaction history
        model: The OpenAI model to use for feedback
        max_tool_response_length: Maximum length for tool responses before truncation
        
    Returns:
        Feedback from the LLM
    """
    if mode == 'history':
        formatted_res = format_history_for_llm(history_or_trajectory, max_tool_response_length=max_tool_response_length)
    else:
        formatted_res = format_trajectory_for_llm(history_or_trajectory, max_tool_response_length=max_tool_response_length)
    
    # Prepare the prompt for the LLM
    prompt: list[dict[str, str]] = [
        {"role": "system", "content": """You are an expert at analyzing AI agent trajectories. 
        Review the following agent interaction history and provide feedback on:
        1. Whether tool responses match tool calls
        2. If the progression of steps looks logical and reasonable
        3. Any errors that appear to be system issues rather than agent mistakes
        4. Overall assessment of the agent's performance

        It's not your job to judge whether the agent's proposed code changes are correct or not; that will be evaluated by unit tests.
        However, if you notice Any obvious mistakes in the agent's reasoning or process, you may note those.
        Your main focus, though, is on checking the validity of the agent framework itself, rather than the intelligence of the underlying model.

        One rule the agent should be following is to issue only one action per turn, either in the form of a tool call or in a <command></command> block.
        If the agent isn't following this rule, that will be the source of issues.
        
        Be specific in your analysis and provide examples from the trajectory to support your observations."""}
    ]
    
    # Add a user message with the formatted history
    prompt.append({"role": "user", "content": f"Here is the agent trajectory to analyze:\n{json.dumps(formatted_res, indent=2)}\n\nPlease provide your expert feedback on this trajectory."})

    client = OpenAI()
    
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        return response.choices[0].message.content or "No feedback generated"
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

def render_history_step(step: dict[str, Any], step_num: int, include_reasoning: bool = True) -> str:
    """
    Render a single history step as a string.
    
    Args:
        step: Single history step
        step_num: Step number for display
        include_reasoning: Whether to include reasoning in the output
        
    Returns:
        Formatted string representation of the step
    """
    from rich.markup import escape
    
    lines = []
    lines.append(f"Step {step_num}:")
    
    # Print role
    if "role" in step:
        lines.append(f"Role: {step['role']}")
    
    # Print content with proper formatting
    if "content" in step:
        content = escape(str(step["content"]))
        lines.append(f"Content: {content}")
    
    # Print any other fields that might be present
    for key, value in step.items():
        if key == 'reasoning' and isinstance(value, list):
            for r in value:
                if isinstance(r, dict) and 'encrypted_content' in r:
                    del r['encrypted_content']
        if key not in ["role", "content"] and (include_reasoning or key != 'reasoning'):
            lines.append(f"{key}: {escape(str(value))}")
    
    return "\n".join(lines)

def render_trajectory_step(step: dict[str, Any], step_num: int, include_reasoning: bool = True) -> str:
    """
    Render a single trajectory step as a string.
    
    Args:
        step: Single trajectory step
        step_num: Step number for display
        include_reasoning: Whether to include reasoning in the output
        
    Returns:
        Formatted string representation of the step
    """
    from rich.markup import escape
    
    lines = []
    lines.append(f"Step {step_num}:")
    
    # Print response
    if "response" in step:
        response = escape(str(step["response"]))
        lines.append(f"Response: {response}")
    
    # Print any other fields that might be present
    for key, value in step.items():
        if key == 'reasoning' and isinstance(value, list):
            for r in value:
                if isinstance(r, dict) and 'encrypted_content' in r:
                    del r['encrypted_content']
        if key != 'response' and (include_reasoning or key != 'reasoning'):
            lines.append(f"{key}: {escape(str(value))}")
    
    return "\n".join(lines)

def get_assistant_steps_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract only assistant steps from history for step alignment.
    
    Args:
        history: List of history items
        
    Returns:
        List of assistant steps only
    """
    if isinstance(history, str):
        history = json.loads(history)
    
    assistant_steps = []
    for step in history:
        if step.get('role') == 'assistant':
            assistant_steps.append(step)
    
    return assistant_steps

def print_side_by_side(answers: dict[str, dict[str, Any]], console: Console, include_reasoning: bool = True, feedback_args: dict[str, Any] | None = None) -> None:
    """
    Print trajectories/histories side by side using a clean table format.
    
    Args:
        answers: Dictionary mapping model names to their answer objects
        console: Rich console for output
        include_reasoning: Whether to include reasoning in the output
        feedback_args: Optional arguments for feedback generation
    """
    if len(answers) == 1:
        # Single model - just use existing functions normally
        model, answer = next(iter(answers.items()))
        console.print(f"\n[bold]Run ID:[/bold] {answer['run_id']}")
        
        log_path = Path("agentic_code_runner/data/trajectories/" + answer['run_id'] + "/" + answer['question_id'] + "/" + answer['question_id'] + ".debug.log")
        if log_path.exists():
            console.print(f"[bold]Log Path:[/bold] {log_path.resolve()}")
        
        if "history" in answer:
            if not feedback_args or not feedback_args.get('feedback_only'):
                console.print(f"\n[bold]History for {model}:[/bold]")
                print_history(answer["history"], console, include_reasoning=include_reasoning)
            
            if feedback_args and feedback_args.get('generate_feedback'):
                console.print(f"\n[bold]Generating LLM feedback on {model} trajectory...[/bold]")
                feedback = generate_llm_feedback(
                    answer["history"], 
                    model=feedback_args['feedback_model'],
                    max_tool_response_length=feedback_args['max_tool_response_length'],
                    mode='history'
                )
                console.print(Panel(feedback, title=f"LLM Feedback for {model}", expand=False))
        elif "trajectory" in answer:
            if not feedback_args or not feedback_args.get('feedback_only'):
                console.print(f"\n[bold]Trajectory for {model}:[/bold]")
                print_trajectory(answer["trajectory"], console, include_reasoning=include_reasoning)
            
            if feedback_args and feedback_args.get('generate_feedback'):
                console.print(f"\n[bold]Generating LLM feedback on {model} trajectory...[/bold]")
                feedback = generate_llm_feedback(
                    answer["trajectory"], 
                    model=feedback_args['feedback_model'],
                    max_tool_response_length=feedback_args['max_tool_response_length'],
                    mode='trajectory'
                )
                console.print(Panel(feedback, title=f"LLM Feedback for {model}", expand=False))
        else:
            console.print(f"[bold red]Missing history/trajectory for {model}, question ID: {answer['question_id']}, answer id: {answer['answer_id']}[/bold red]")
    else:
        # Multiple models - create clean side by side comparison
        model_names = list(answers.keys())
        
        # Print run IDs and log paths first
        for model, answer in answers.items():
            console.print(f"[bold]{model} Run ID:[/bold] {answer['run_id']}")
            log_path = Path("agentic_code_runner/data/trajectories/" + answer['run_id'] + "/" + answer['question_id'] + "/" + answer['question_id'] + ".debug.log")
            if log_path.exists():
                console.print(f"[bold]{model} Log Path:[/bold] {log_path.resolve()}")
        
        console.print()  # Add spacing
        
        # Extract data for both models
        model_data = {}
        mode = None
        
        for model, answer in answers.items():
            if "history" in answer:
                if mode is None:
                    mode = 'history'
                elif mode != 'history':
                    console.print("[bold red]Error: Cannot mix history and trajectory modes[/bold red]")
                    return
                
                history = answer["history"]
                if isinstance(history, str):
                    history = json.loads(history)
                
                # Extract only assistant steps for alignment
                assistant_steps = []
                for step in history:
                    if step.get('role') == 'assistant':
                        assistant_steps.append(step)
                model_data[model] = assistant_steps
                
            elif "trajectory" in answer:
                if mode is None:
                    mode = 'trajectory'
                elif mode != 'trajectory':
                    console.print("[bold red]Error: Cannot mix history and trajectory modes[/bold red]")
                    return
                
                trajectory = answer["trajectory"]
                if isinstance(trajectory, str):
                    trajectory = json.loads(trajectory)
                model_data[model] = trajectory
            else:
                console.print(f"[bold red]Missing history/trajectory for {model}[/bold red]")
                model_data[model] = []
        
        if not feedback_args or not feedback_args.get('feedback_only'):
            # Find the maximum number of steps
            max_steps = max(len(steps) for steps in model_data.values()) if model_data else 0
            
            if max_steps == 0:
                console.print("[bold red]No steps found in any model[/bold red]")
                return
            
            # Create table
            table = Table(show_header=True, expand=True, box=box.ROUNDED)
            table.add_column("Step", style="bold cyan", width=8)
            for model in model_names:
                table.add_column(model, ratio=1)
            
            # Add rows for each step
            for step_idx in range(max_steps):
                row_data = [f"Step {step_idx + 1}"]
                
                for model in model_names:
                    steps = model_data[model]
                    
                    if step_idx < len(steps):
                        step = steps[step_idx]
                        
                        # Format step data cleanly
                        step_parts = []
                        
                        if mode == 'history':
                            # For history mode, show key fields
                            if "content" in step:
                                content = str(step["content"])
                                step_parts.append(f"[bold]Content:[/bold]\n{escape(content)}")
                            
                            # Show tool calls if present
                            if "tool_calls" in step and step["tool_calls"]:
                                tool_calls = step["tool_calls"]
                                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                                    tool_call = tool_calls[0]
                                    if "function" in tool_call:
                                        func_name = tool_call["function"].get("name", "unknown")
                                        step_parts.append(f"[bold]Tool Call:[/bold] {func_name}")
                            
                            # Show other relevant fields
                            for key, value in step.items():
                                if key not in ["role", "content", "tool_calls"] and (include_reasoning or key != 'reasoning'):
                                    if key == 'reasoning' and isinstance(value, list):
                                        # Clean up reasoning
                                        for r in value:
                                            if isinstance(r, dict) and 'encrypted_content' in r:
                                                del r['encrypted_content']
                                    
                                    value_str = str(value)
                                    step_parts.append(f"[bold]{key}:[/bold] {escape(value_str)}")
                        
                        else:  # trajectory mode
                            # For trajectory mode, show action, thought, observation
                            if "action" in step:
                                action = str(step["action"])
                                step_parts.append(f"[bold]Action:[/bold]\n{escape(action)}")
                            
                            if "thought" in step:
                                thought = str(step["thought"])
                                step_parts.append(f"[bold]Thought:[/bold]\n{escape(thought)}")
                            
                            if "observation" in step:
                                observation = str(step["observation"])
                                step_parts.append(f"[bold]Observation:[/bold]\n{escape(observation)}")
                            
                            # Show other fields
                            for key, value in step.items():
                                if key not in ["action", "thought", "observation"] and (include_reasoning or key != 'reasoning'):
                                    if key == 'reasoning' and isinstance(value, list):
                                        # Clean up reasoning
                                        for r in value:
                                            if isinstance(r, dict) and 'encrypted_content' in r:
                                                del r['encrypted_content']
                                    
                                    value_str = str(value)
                                    step_parts.append(f"[bold]{key}:[/bold] {escape(value_str)}")
                        
                        row_data.append("\n\n".join(step_parts))
                    else:
                        # No step for this model at this index
                        row_data.append("[dim]No step[/dim]")
                
                # Add row with end_section=True to create a line after each step (except the last)
                table.add_row(*row_data, end_section=(step_idx < max_steps - 1))
            
            # Display the table
            console.print(table)
        
        # Generate feedback if requested
        if feedback_args and feedback_args.get('generate_feedback'):
            console.print("\n[bold]Generating LLM feedback...[/bold]")
            
            for model, answer in answers.items():
                if "history" in answer:
                    feedback = generate_llm_feedback(
                        answer["history"], 
                        model=feedback_args['feedback_model'],
                        max_tool_response_length=feedback_args['max_tool_response_length'],
                        mode='history'
                    )
                elif "trajectory" in answer:
                    feedback = generate_llm_feedback(
                        answer["trajectory"], 
                        model=feedback_args['feedback_model'],
                        max_tool_response_length=feedback_args['max_tool_response_length'],
                        mode='trajectory'
                    )
                else:
                    feedback = "No history/trajectory available for feedback"
                
                console.print(Panel(feedback, title=f"LLM Feedback for {model}", expand=False))

def main():
    parser = argparse.ArgumentParser(description="Inspect agentic trajectories for a specific model and question ID.")
    parser.add_argument("--model", required=True, nargs='+', help="Name(s) of the model(s) to inspect (1-2 models)")
    parser.add_argument("--question-id", required=True, help="ID of the question to inspect")
    parser.add_argument("--generate-feedback", action="store_true", help="Generate LLM feedback on the agent trajectory")
    parser.add_argument("--feedback-model", default="gpt-4-turbo", help="Model to use for generating feedback (default: gpt-4-turbo)")
    parser.add_argument("--max-tool-response-length", type=int, default=1000, help="Maximum length for tool responses before truncation (default: 1000)")
    parser.add_argument("--feedback-only", action="store_true", help="Only generate feedback without inspecting the trajectory")
    parser.add_argument("--no-include-reasoning", action="store_true", help="Don't include model reasoning output")
    args = parser.parse_args()
    
    # Validate number of models
    if len(args.model) > 2:
        print("Error: Maximum of 2 models can be compared at once.")
        sys.exit(1)
    
    # Create rich console
    console = Console()
    
    # Load answers for each model
    model_answers = {}
    for model in args.model:
        console.print(f"Loading answers for model [bold]{model}[/bold]...")
        model_name = get_model_config(model).display_name
        answers = load_model_answers(model_name)
        console.print(f"Loaded {len(answers)} answers for {model}.")
        
        # Find answer with the specified question ID
        answer = find_answer_by_question_id(answers, args.question_id)
        if answer:
            model_answers[model] = answer
        else:
            console.print(f"No answer found for question ID {args.question_id} from model {model}", style="bold red")
    
    if not model_answers:
        console.print(f"No answers found for question ID {args.question_id} from Any of the specified models", style="bold red")
        sys.exit(1)
    
    # Print question ID and model names
    console.print(f"\n[bold]Question ID:[/bold] {args.question_id}")
    console.print(f"[bold]Model(s):[/bold] {', '.join(model_answers.keys())}")
    
    # Prepare feedback arguments if needed
    feedback_args = None
    if args.generate_feedback or args.feedback_only:
        feedback_args = {
            'generate_feedback': args.generate_feedback,
            'feedback_only': args.feedback_only,
            'feedback_model': args.feedback_model,
            'max_tool_response_length': args.max_tool_response_length
        }
    
    # Use the side-by-side function
    print_side_by_side(model_answers, console, include_reasoning=not args.no_include_reasoning, feedback_args=feedback_args)

if __name__ == "__main__":
    main()