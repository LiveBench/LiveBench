# LiveBench Agentic Coding Evaluation

This directory contains the evaluation code for the LiveBench agentic coding task, which consists of testing models on their ability to resolve [Multi-SWE-Bench](https://multi-swe-bench.github.io/#/)-style tasks under the [SWE-Agent](https://swe-agent.com) agent framework.

The code here is adapted from the Multi-SWE-Bench [GitHub repository](https://github.com/multi-swe-bench/multi-swe-bench) with minimal modifications to ensure it can be triggered from the broader LiveBench evaluation harness.

The main modification that has been made is to update the Docker image build process to ensure all instance images have Python, pip, and pipx installed and a virtual environment activated, as these are necessary in order for SWE-Rex (the backend of SWE-Agent) to function in the Docker container.

The LICENSE file from the Multi-SWE-Bench GitHub repository is included here. (original source https://github.com/multi-swe-bench/multi-swe-bench/blob/main/LICENSE)