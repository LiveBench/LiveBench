"""Replay agent that replays actions from a saved trajectory."""

import json
from pathlib import Path
from typing import Callable

from livebench.agentic_code_runner.minisweagent import Environment, Model
from livebench.agentic_code_runner.minisweagent.agents.default import AgentConfig, DefaultAgent
from livebench.agentic_code_runner.minisweagent.utils.log import logger


class ReplayAgent(DefaultAgent):
    """Agent that replays actions from a saved trajectory file.
    
    Instead of querying the model, this agent uses pre-recorded responses
    from a trajectory file to execute actions in sequence.
    """

    def __init__(self, model: Model, env: Environment, trajectory_path: str | Path, *, config_class: Callable = AgentConfig, **kwargs):
        """Initialize the replay agent.
        
        Args:
            model: The model (used for compatibility, but not queried)
            env: The environment to execute actions in
            trajectory_path: Path to the trajectory file to replay
            **kwargs: Additional arguments passed to DefaultAgent
        """
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.trajectory_path = Path(trajectory_path)
        self.trajectory_messages: list[dict] = []
        self.current_message_idx = 0
        self._load_trajectory()

    def _load_trajectory(self):
        """Load the trajectory file and extract assistant messages."""
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_path}")
        
        with open(self.trajectory_path) as f:
            trajectory_data = json.load(f)
        
        if "messages" not in trajectory_data:
            raise ValueError(f"Invalid trajectory file: no 'messages' field in {self.trajectory_path}")
        
        # Extract only assistant messages (those are the ones we need to replay)
        self.trajectory_messages = [
            msg for msg in trajectory_data["messages"] 
            if msg.get("role") == "assistant"
        ]
        
        logger.info(f"Loaded trajectory with {len(self.trajectory_messages)} assistant messages from {self.trajectory_path}")

    def query(self) -> dict:
        """Return the next pre-recorded response instead of querying the model.
        Turns into DefaultAgent once all messages have been replayed.
        
        Returns:
            The next assistant message from the trajectory
        """
        if self.current_message_idx >= len(self.trajectory_messages):
            res = super().query()
            logger.info("Actual query " + str(self.model.n_calls))
            # logger.info(res['content'])
            return res
        
        # Get the next message from the trajectory
        message = self.trajectory_messages[self.current_message_idx]
        if message.get("role") != "assistant":
            self.current_message_idx += 1
            message = self.trajectory_messages[self.current_message_idx]
        self.current_message_idx += 1
        
        logger.info(f"Replaying message {self.current_message_idx}/{len(self.trajectory_messages)}")
        
        # logger.info(message['content'])
        
        # Add the message to our messages list (so the trajectory is consistent)
        self.add_message(**message)
        self.model.n_calls += 1
        
        # Return the message in the same format as Model.query() would
        return message

    def get_observation(self, response: dict) -> dict:
        output = super().get_observation(response)
        # observation = self.render_template(self.config.action_observation_template, output=output)
        # logger.info("Got Observation")
        # print(observation)
        return output