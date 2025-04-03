import json
import argparse
from collections import defaultdict

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_difficulties_and_scores(question_files, score_files, model, score_threshold=0.5):
    # Load and combine all question data
    questions = []
    for qfile in question_files:
        questions.extend(load_jsonl(qfile))
    
    print(len(questions))

    # Load and combine all score data
    scores = []
    for sfile in score_files:
        scores.extend(load_jsonl(sfile))

    print(len(scores))

    # Create question_id to difficulty mapping
    # question_difficulty = {q['question_id']: q['original_json']['difficulty'] for q in questions}
    # question_difficulty = {q['question_id']: q['release_date'][:-3] for q in questions}
    question_difficulty = {q['question_id']: q['question_type'] for q in questions}
    release_dates = {q['livebench_release_date'] for q in questions}

    # Initialize counters for right/wrong per difficulty
    difficulty_results = defaultdict(lambda: {'right': 0, 'wrong': 0})

    # Count frequencies
    for score_entry in scores:
        question_id = score_entry['question_id']
        score = score_entry['score']
        difficulty = question_difficulty.get(question_id)

        if score_entry['model'] != model:
            continue

        if difficulty:  # Only process if we have difficulty info
            if score >= score_threshold:
                difficulty_results[difficulty]['right'] += 1
            else:
                difficulty_results[difficulty]['wrong'] += 1

    # Print results
    print(f"Model: {model.replace('-high', '')}")
    print(f"Livebench release: {max(release_dates)}")
    print("\nResults by difficulty level:")
    print("-" * 50)
    for difficulty, counts in sorted(difficulty_results.items()):
        total = counts['right'] + counts['wrong']
        accuracy = (counts['right'] / total * 100) if total > 0 else 0
        print(f"Difficulty: {difficulty}")
        print(f"Right: {counts['right']}, Wrong: {counts['wrong']}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Analyze question difficulties and scores from JSONL files.')
    parser.add_argument('--question-files', nargs='+', required=True,
                      help='List of JSONL files containing questions')
    parser.add_argument('--score-files', nargs='+', required=True,
                      help='List of JSONL files containing scores')
    parser.add_argument('--model', type=str)
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Score threshold for considering an answer correct (default: 0.5)')

    args = parser.parse_args()

    analyze_difficulties_and_scores(
        args.question_files,
        args.score_files,
        args.model,
        args.threshold
    )

if __name__ == "__main__":
    main()