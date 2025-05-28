# LiveBench Agentic Coding Agent

This directory contains a version of [SWE-Agent](https://swe-agent.com) adapted for use in LiveBench. All models are queried under this framework to solve real-world coding problems.

The SWE-Agent code is pulled from the [github repository](https://github.com/SWE-agent/SWE-agent/tree/main). Various modifications have been made and unnecessary features and scripts have been removed.

The `run_agentic_coding_inference` function in `run_inference.py` takes in a LiveBench question object and prepares the command to execute SWE-Agent on that question. Docker containers from Multi-SWE-Bench are used.