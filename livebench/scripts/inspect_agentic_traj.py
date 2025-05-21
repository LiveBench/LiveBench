#!/usr/bin/env python3
"""
Inspect agentic trajectories for a specific model and question ID.

Usage:
python inspect_agentic_traj.py --model <model_name> --question-id <question_id> [--generate-feedback] [--feedback-model <model_name>] [--max-tool-response-length <length>]

Options:
  --model: Name of the model to inspect (required)
  --question-id: ID of the question to inspect (required)
  --generate-feedback: Generate LLM feedback on the agent trajectory (optional)
  --feedback-model: Model to use for generating feedback (default: gpt-4-turbo)
  --max-tool-response-length: Maximum length for tool responses before truncation (default: 1000)
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Literal
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from livebench.model.api_model_config import get_model_config

def load_model_answers(model_name: str) -> List[Dict[str, Any]]:
    """
    Load model answers from the specified JSONL file.
    
    Args:
        model_name: Name of the model whose answers to load
        
    Returns:
        List of answer objects from the JSONL file
    """
    file_path = f"data/live_bench/coding/agentic_coding/model_answer/{model_name}.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find answer file for model {model_name} at {file_path}.")
        sys.exit(1)
    
    answers = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                answer_obj = json.loads(line.strip())
                answers.append(answer_obj)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}. Skipping.")
                continue
    
    return answers

def find_answer_by_question_id(answers: List[Dict[str, Any]], question_id: str) -> Dict[str, Any]:
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

def print_history(history: List[Dict[str, Any]], console: Console, include_reasoning: bool = True) -> None:
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
        
        # Print any other fields that might be present
        for key, value in step.items():
            if key == 'reasoning' and isinstance(value, list):
                for r in value:
                    if isinstance(r, dict) and 'encrypted_content' in r:
                        del r['encrypted_content']
            if key not in ["role", "content"] and (include_reasoning or key != 'reasoning'):
                console.print(f"[bold cyan]{key}:[/bold cyan] {escape(str(value))}")
        
        if step['role'] == 'assistant':
                i += 1

def print_trajectory(trajectory: List[Dict[str, Any]], console: Console, include_reasoning: bool = True) -> None:
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
        
        # Print any other fields that might be present
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

def format_history_for_llm(history: List[Dict[str, Any]], max_tool_response_length: int = 1000) -> List[Dict[str, Any]]:
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

def format_trajectory_for_llm(trajectory: List[Dict[str, Any]], max_tool_response_length: int = 1000) -> List[Dict[str, Any]]:
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

def generate_llm_feedback(history_or_trajectory: List[Dict[str, Any]], model: str = "gpt-4.1", max_tool_response_length: int = 1000, mode: Literal['history', 'trajectory'] = 'history') -> str:
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
    prompt = [
        {"role": "system", "content": """You are an expert at analyzing AI agent trajectories. 
        Review the following agent interaction history and provide feedback on:
        1. Whether tool responses match tool calls
        2. If the progression of steps looks logical and reasonable
        3. Any errors that appear to be system issues rather than agent mistakes
        4. Overall assessment of the agent's performance

        It's not your job to judge whether the agent's proposed code changes are correct or not; that will be evaluated by unit tests.
        However, if you notice any obvious mistakes in the agent's reasoning or process, you may note those.
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
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Inspect agentic trajectories for a specific model and question ID.")
    parser.add_argument("--model", required=True, help="Name of the model to inspect")
    parser.add_argument("--question-id", required=True, help="ID of the question to inspect")
    parser.add_argument("--generate-feedback", action="store_true", help="Generate LLM feedback on the agent trajectory")
    parser.add_argument("--feedback-model", default="gpt-4-turbo", help="Model to use for generating feedback (default: gpt-4-turbo)")
    parser.add_argument("--max-tool-response-length", type=int, default=1000, help="Maximum length for tool responses before truncation (default: 1000)")
    parser.add_argument("--feedback-only", action="store_true", help="Only generate feedback without inspecting the trajectory")
    parser.add_argument("--no-include-reasoning", action="store_true", help="Don't include model reasoning output")
    args = parser.parse_args()
    
    # Create rich console
    console = Console()
    
    # Load model answers
    console.print(f"Loading answers for model [bold]{args.model}[/bold]...")
    model_name = get_model_config(args.model).display_name
    answers = load_model_answers(model_name)
    console.print(f"Loaded {len(answers)} answers.")
    
    # Find answer with the specified question ID
    console.print(f"Finding answer for question ID [bold]{args.question_id}[/bold]...")
    answer = find_answer_by_question_id(answers, args.question_id)
    
    if not answer:
        console.print(f"No answer found for question ID {args.question_id} from model {args.model}", style="bold red")
        sys.exit(1)
    
    # Print question ID and model name
    console.print(f"\n[bold]Question ID:[/bold] {args.question_id}")
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"\n[bold]Run ID:[/bold] {answer['run_id']}")

    log_path = Path("agentic_code_runner/data/trajectories/" + answer['run_id'] + "/" + answer['question_id'] + "/" + answer['question_id'] + ".debug.log")
    if log_path.exists():
        console.print(f"\n[bold]Log Path:[/bold] {log_path.resolve()}")
    

    if "history" in answer:
        if not args.feedback_only:
            # Print history field step by step
            console.print("\n[bold]History:[/bold]")
            print_history(answer["history"], console, include_reasoning=not args.no_include_reasoning)
        
        # Generate feedback if requested
        if args.generate_feedback:
            console.print("\n[bold]Generating LLM feedback on trajectory...[/bold]")
            feedback = generate_llm_feedback(
                answer["history"], 
                model=args.feedback_model,
                max_tool_response_length=args.max_tool_response_length,
                mode='history'
            )
            console.print(Panel(feedback, title="LLM Feedback", expand=False))
    elif "trajectory" in answer:
        if not args.feedback_only:
            # Print history field step by step
            console.print("\n[bold]Trajectory:[/bold]")
            print_trajectory(answer["trajectory"], console, include_reasoning=not args.no_include_reasoning)
        
        # Generate feedback if requested
        if args.generate_feedback:
            console.print("\n[bold]Generating LLM feedback on trajectory...[/bold]")
            feedback = generate_llm_feedback(
                answer["trajectory"], 
                model=args.feedback_model,
                max_tool_response_length=args.max_tool_response_length,
                mode='trajectory'
            )
            console.print(Panel(feedback, title="LLM Feedback", expand=False))
    else:
        raise ValueError(f"Missing history for question ID: {args.question_id}, answer id: {answer['answer_id']}")

if __name__ == "__main__":
    main()