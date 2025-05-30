# LiveBench Agentic Coding Agent

This directory contains a version of [SWE-Agent](https://swe-agent.com) adapted for use in LiveBench. All models are queried under this framework to solve real-world coding problems.

The SWE-Agent code is pulled from the [github repository](https://github.com/SWE-agent/SWE-agent/tree/main). Various modifications have been made and unnecessary features and scripts have been removed.

The `run_agentic_coding_inference` function in `run_inference.py` takes in a LiveBench question object and prepares the command to execute SWE-Agent on that question. Docker containers from Multi-SWE-Bench are used.

## Function Calling Configuration

Model configurations in livebench.model.model_configs are used to determine various configuration parameters for SWE-Agent, including max_input_tokens, max_output_tokens, and supports_function_calling. Information from [litellm](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) is also used, if no agent_config is present for a given model, and defines the same attributes. If the model supports function calling, then it is used for tool calls; if not, or if we found that the function calling support is not very functional, then the XMLThoughtAction parser is used, where models output their commands in `<command></command>` blocks.