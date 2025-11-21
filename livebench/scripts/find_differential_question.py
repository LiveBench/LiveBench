import json
from collections import defaultdict
import argparse
from pathlib import Path
from livebench.model.api_model_config import get_model_config
from rich.console import Console
from rich.table import Table

def find_differential_problems(filename, models):
    # Dictionary to store scores for each question_id and model
    scores = defaultdict(dict)

    # Read the JSONL file and process each line
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = data['question_id']
            model = data['model']
            score = data['score']
            
            if model in models:
                scores[qid][model] = score

    # Find questions where models have different outcomes (some correct, some incorrect)
    differential_cases = []
    for qid, model_scores in scores.items():
        if len(model_scores) == len(models):  # ensure we have scores for all models
            correct_models = [model for model in models if model_scores[model] > 0]
            incorrect_models = [model for model in models if model_scores[model] == 0]
            
            # Only include if there are both correct and incorrect models (i.e., not all same)
            if correct_models and incorrect_models:
                differential_cases.append((qid, correct_models, incorrect_models))

    return differential_cases

def print_table(cases, console):
    if not cases:
        console.print("\n[yellow]No differential cases found.[/yellow]")
        return
    
    table = Table(title=f"Differential Results ({len(cases)} questions)", show_lines=True)
    
    table.add_column("Question ID", style="cyan", no_wrap=True)
    table.add_column("Correct Models", style="green")
    table.add_column("Incorrect Models", style="red")
    
    for qid, correct, incorrect in cases:
        correct_str = ", ".join(correct).replace("11-2025", "")
        incorrect_str = ", ".join(incorrect).replace("11-2025", "")
        table.add_row(qid, correct_str, incorrect_str)
    
    console.print()
    console.print(table)

def process_root_directory(root_dir, models):
    console = Console()
    root_path = Path(root_dir)
    
    # Find all ground_truth_judgment.jsonl files recursively
    judgment_files = list(root_path.rglob('ground_truth_judgment.jsonl'))
    
    if not judgment_files:
        console.print(f"[red]No ground_truth_judgment.jsonl files found in {root_dir}[/red]")
        return
    
    console.print(f"[bold]Found {len(judgment_files)} ground_truth_judgment.jsonl file(s)[/bold]")
    console.print(f"[bold]Comparing {len(models)} models:[/bold] {', '.join(models)}\n")
    console.rule("[bold blue]Results[/bold blue]")
    
    # Process each file
    for judgment_file in sorted(judgment_files):
        differential_cases = find_differential_problems(judgment_file, models)
        
        # Print results for this file
        console.print()
        console.rule(f"[bold cyan]{judgment_file.relative_to(root_path)}[/bold cyan]")
        
        print_table(differential_cases, console)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find problems where models have different outcomes (some correct, some incorrect)'
    )
    parser.add_argument('root_dir', help='Root directory to search for ground_truth_judgment.jsonl files')
    parser.add_argument('models', nargs='+', help='Model names to compare (2 or more)')
    
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("At least 2 models are required for comparison")

    # Get display names for all models
    model_display_names = [get_model_config(model).display_name for model in args.models]
    
    process_root_directory(args.root_dir, model_display_names)