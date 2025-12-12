import argparse
import glob
import json
import os
import time
from pathlib import Path

import shortuuid
import yaml

from livebench.common import (
    LIVE_BENCH_RELEASES,
    LIVE_BENCH_ROOT_PATH,
    load_questions_jsonl,
    reorg_answer_file,
)
from livebench.model import get_model_config
from livebench.model.completions import API_ERROR_OUTPUT


def _split_model_name(model_name: str) -> tuple[str, str]:
    if "/" in model_name:
        provider, api_name = model_name.split("/", 1)
    else:
        provider, api_name = "", model_name
    return provider, api_name


def _load_questions(model_display_name: str, release_option: str) -> tuple[list[dict], set[str]]:
    if release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {release_option}.")

    release_set = {r for r in LIVE_BENCH_RELEASES if r <= release_option}

    base = LIVE_BENCH_ROOT_PATH / "data/live_bench/agentic_coding"
    question_files = glob.glob(str(base / "**/question.jsonl"), recursive=True)

    questions: list[dict] = []
    answer_files: set[str] = set()

    for question_file in question_files:
        qlist = load_questions_jsonl(
            question_file,
            release_set,
            release_option,
            None,
        )
        bench_name_for_file = os.path.dirname(question_file).replace(
            f"{LIVE_BENCH_ROOT_PATH}/data/", ""
        )
        answer_file = (
            LIVE_BENCH_ROOT_PATH
            / "data"
            / bench_name_for_file
            / "model_answer"
            / f"{model_display_name.lower()}.jsonl"
        )

        for q in qlist:
            q["answer_file"] = str(answer_file)
        questions.extend(qlist)
        answer_files.add(str(answer_file))

    return questions, answer_files


def _build_answer_base(
    question_id: str,
    run_id: str,
    model_display_name: str,
    provider: str,
    api_name: str,
    api_kwargs: dict,
) -> dict:
    return {
        "question_id": question_id,
        "answer_id": shortuuid.uuid(),
        "run_id": run_id,
        "model_id": model_display_name,
        "tstamp": time.time(),
        "api_info": {
            "provider": provider,
            "api_name": api_name,
            "api_kwargs": api_kwargs,
        },
    }


def _write_answer(answer_file: str, ans: dict) -> None:
    path = Path(answer_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans) + "\n")


def _extract_tokens(info: dict) -> tuple[int | None, int | None]:
    stats = info.get("model_stats", {}) if isinstance(info, dict) else {}
    return stats.get("total_output_tokens"), stats.get("total_input_tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescue agentic coding submissions into answer jsonl",
    )
    parser.add_argument("--model", required=True, help="Model name to resolve display name")
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Path to trajectory run directory (e.g., agentic_code_runner/data/trajectories/<run_id>)",
    )
    parser.add_argument(
        "--livebench-release",
        default=max(LIVE_BENCH_RELEASES),
        help="LiveBench release to use when loading questions",
    )

    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    run_id = traj_dir.name
    config_path = traj_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {traj_dir}")

    with config_path.open(encoding="utf-8") as f:
        traj_config = yaml.safe_load(f)

    config_model = traj_config.get("model", {}) if isinstance(traj_config, dict) else {}
    model_name_raw = config_model.get("model_name", "")
    api_kwargs = config_model.get("model_kwargs") or {}
    provider, api_name = _split_model_name(model_name_raw)

    model_display_name = get_model_config(args.model).display_name

    questions, answer_files = _load_questions(model_display_name, args.livebench_release)
    if not questions:
        print("No questions loaded; nothing to do.")
        return

    processed = 0
    missing = 0

    for question in questions:
        qid = str(question["question_id"])
        traj_path = traj_dir / qid / f"{qid}.traj.json"

        ans = _build_answer_base(
            qid,
            run_id,
            model_display_name,
            provider,
            api_name,
            api_kwargs,
        )

        if traj_path.exists():
            trajectory = json.load(traj_path.open())
            info = trajectory.get("info", {}) if isinstance(trajectory, dict) else {}
            final_answer = info.get("submission")
            if final_answer is None:
                final_answer = ""
            if "submission" in info:
                info.pop("submission", None)
            trajectory["info"] = info

            total_output_tokens, total_input_tokens = _extract_tokens(info)

            ans.update(
                {
                    "trajectory": json.dumps(trajectory, indent=4),
                    "choices": [{"turns": [final_answer]}],
                    "total_output_tokens": total_output_tokens,
                    "total_input_tokens": total_input_tokens,
                }
            )
        else:
            ans["choices"] = [{"turns": [API_ERROR_OUTPUT]}]
            missing += 1

        answer_file = question["answer_file"]
        _write_answer(answer_file, ans)
        processed += 1

    for answer_file in answer_files:
        reorg_answer_file(answer_file)

    print(
        f"Wrote {processed} answers ({missing} missing trajectories) to {len(answer_files)} file(s)."
    )


if __name__ == "__main__":
    main()

