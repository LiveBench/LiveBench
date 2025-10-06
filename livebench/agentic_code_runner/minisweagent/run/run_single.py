"""Run on a single SWE-Bench instance."""

import json
import traceback
from pathlib import Path

import typer
import yaml

from livebench.agentic_code_runner.minisweagent.agents.interactive import InteractiveAgent
from livebench.agentic_code_runner.minisweagent.agents.replay import ReplayAgent
from livebench.agentic_code_runner.minisweagent.config import get_config_path
from livebench.agentic_code_runner.minisweagent.models import get_model
from livebench.agentic_code_runner.minisweagent.run.run_batch import (
    get_sb_environment,
)
from livebench.agentic_code_runner.minisweagent.run.utils.save import save_traj
from livebench.agentic_code_runner.minisweagent.utils.log import add_file_handler, logger

app = typer.Typer(add_completion=False)


# fmt: off
@app.command()
def main(
    config_path: Path = typer.Option( None, "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    base_output_path: Path = typer.Option(None, "-o", "--output", help="Output trajectory Directory", rich_help_panel="Basic"),
    instances_path: Path = typer.Option(None, "-i", "--instances_path", help="Path to a instances file", rich_help_panel="Basic"),
    replay_traj: Path | None = typer.Option(None, "-r", "--replay-traj", help="Path to trajectory file to replay (replay mode)", rich_help_panel="Replay"),
) -> None:
    # fmt: on
    """Run on a single SWE-Bench instance."""
    instances = [json.loads(line) for line in open(instances_path)]
    if len(instances) == 0:
        logger.info("No instances found")
        return
    elif len(instances) > 1:
        logger.info("More than one instance found, only running on the first one")
    instance = instances[0]

    output_path = Path(base_output_path) / instance['instance_id']
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    config_path = get_config_path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if replay_traj is None:
        config.setdefault("agent", {})["confirm_exit"] = False
    env = get_sb_environment(config, instance)
    
    # Check if we're in replay mode
    if replay_traj is not None:
        logger.info(f"Replay mode: loading trajectory from {replay_traj}")
        agent = ReplayAgent(
            get_model(config=config.get("model", {})),
            env,
            trajectory_path=replay_traj,
            **config.get("agent", {}),
        )
    else:
        agent = InteractiveAgent(
            get_model(config=config.get("model", {})),
            env,
            **({"mode": "yolo"} | config.get("agent", {})),
        )

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output_path / f"{instance['instance_id']}.traj.json", exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()