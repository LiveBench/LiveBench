#!/usr/bin/env python3

import argparse
import os
import sys


def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser(description="Rename a model across data files.")
    _ = parser.add_argument("old_name", type=str, help="Existing model name to replace")
    _ = parser.add_argument("new_name", type=str, help="New model name to use")
    _ = parser.add_argument(
        "--root",
        type=str,
        default='data',
        help="Root directory to search from (defaults to repository root)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Error: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    return args.old_name, args.new_name, root


def replace_in_file(filepath: str, old_name: str, new_name: str) -> None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return

    if old_name not in content:
        return

    content = content.replace(old_name, new_name)
    with open(filepath, "w", encoding="utf-8") as f:
        _ = f.write(content)


def process_jsonl_files(data_dir: str, old_name: str, new_name: str) -> None:
    for current_root, _dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename == f"{old_name}.jsonl":
                src = os.path.join(current_root, filename)
                # Replace occurrences inside the file first
                replace_in_file(src, old_name, new_name)
                # Then rename the file
                dst = os.path.join(current_root, f"{new_name}.jsonl")
                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)
            elif filename == "ground_truth_judgment.jsonl":
                replace_in_file(os.path.join(current_root, filename), old_name, new_name)


def main() -> None:
    old_name, new_name, root = parse_args()

    if old_name == new_name:
        print("old_name and new_name are identical; nothing to do.")
        return

    process_jsonl_files(root, old_name, new_name)


if __name__ == "__main__":
    main()
