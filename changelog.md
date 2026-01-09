## Changelog

### 2026-01-08
Added a new math task: Integrals with Game
- This version includes a new math task which introduces a game theory problem integraded with integral calculations.
 
### 2025-12-23
Added a new reasoning task: Logic with Navigation
- Tests model to solve a logic problem, and then navigate in the 2D space. 

### 2025-11-25
This update refreshed some existing tasks and updated others to resolve (to some extent) saturation and contamination of the benchmark.
 - Updated connections with new tasks from more recent puzzles
 - Updated math_comp and olympiad with new questions from more recent competitions
 - Updated the instruction following tasks with both newer articles as well as new, more difficult types of instructions, and resolved issues with impossible tasks
 - Updated zebra_puzzle with newer, more difficult puzzles
 - Replaced web_of_lies_v3 with theory_of_mind, a new task to evaluate LLM's ability to reason about the internal states of other people in their environments

### 2025-10-03
Switched the agent used for agentic coding to [Mini-SWE-Agent](https://mini-swe-agent.com/latest/) instead of SWE-Agent due to its simpler design, consistent interface with all models, and easier maintainability. Additionally, we updated the step limit to 250 rather than 50. The agentic coding tasks were rerun for all models with these changes and the existing 2025-05-30 leaderboard was updated with the new results.

### 2025-05-30
This update introduces a new agentic coding category, where models must operate in a multi-turn, realistic development environment to resolve issues from real Github repositories. It contains Python, JavaScript, and TypeScript tasks, and we plan to add more tasks and languages in the future. Inference is performed using the [SWE-Agent](https://swe-agent.com) framework, and evaluation uses the [Multi-SWE-Bench](https://multi-swe-bench.github.io/#/) harness. This task provides the most realistic possible evaluation of LLM coding capabilities and prediction of which will be most useful for developers.

Note: SWE-Agent was run with a 50-step limit for all models during this evaluation. In some cases, it's likely that scores would have improved were models given more time to complete the tasks.

### 2025-04-25
 - Completely new coding questions focused on evaluating usage of real-world libraries in realistic scenarios. Questions are no longer sourced from LiveCodeBench. The tasks themselves are the same; we still have a full code generation task and a code completion task.
 - Refeshed data analysis tasks, specifically for the tablejoin and tablereformat tasks. The cta task has been retired.

### 2025-04-02
 - Refreshed coding questions (coding_completion and LCB_generation) with *much* newer questions from LiveCodeBench. The previous questions were likely heavily contaminated for newer models. LiveCodeBench also increases question difficulty over time.
 - Refreshed typos and plot_unscrambling questions with newer ArXiv papers and movie plots from Wikipedia, respectively. Issues in the typos question generation script were also fixed, so that all questions can be fairly evaluated
 - Replaced 2023 AMC questions with 2024 AMC questions in math_comp
 - Updated web_of_lies with a mix of harder questions of the previous web_of_lies_v2 format and the new format from [BIG-Bench Extra Hard](https://github.com/google-deepmind/bbeh)
 - All new questions ask for answers in the `<solution></solution>` format.

### 2024-11-25
This update focused on refreshing questions to check for contamination and increasing the difficulty of tasks for which o1 (and other reasoning models) achieved very high scores.
 - Refreshed the instruction following tasks with new articles from The Guardian
 - Updated IF question generation to include 3 instructions per task on average (previously 2)
 - Refreshed Connections task with new puzzles from NYT
 - Updated Connections generation to more frequently ask for more groups
 - Regenerated Zebra Puzzles, skewing towards larger board sizes and more complex constraints
 - Updated Connections and Zebra Puzzles questions to require answers to be in `<solution></solution>` tags rather than bolded

### 2024-08-31

This update focused on refreshing question sources for all three math tasks. 
 - A portion of the questions for AMPS_Hard were updated, to increase the difficulty, add new types of questions, and to remove previous ambiguity in the questions.
 - All olympiad questions were fully updated to use IMO 2024 and USAMO 2024 questions as the source, replacing the IMO/USAMO 2023 questions. These questions were also made harder.
 - All 50 AMC 2023 problems were updated to rearrange the multiple choice answer order, and to update the prose and proper nouns in the questions.

### 2024-07-26

This update included new questions for the coding completion and LCB generation tasks. It also introduced a new spatial reasoning task that judges models' ability to reason about 2D and 3D shapes. Examples of questions from this task can be seen below:

 > Suppose I have a physical, solid square with vertices ABCD, and I make two cuts through AC and BD. Of the resulting pieces, how many triangles are there?  Think step by step, and then put your answer in **bold** as a single integer (for example, **0**). If you don't know, guess.

 > Suppose I have three solid spheres of radius 5 resting on a plane. Each sphere is tangent to the other two spheres. Now I add a fourth solid sphere of radius 6 in such a way that maximizes the number of tangent points among all pairs of spheres. If I consider a new shape whose vertices are exactly the centers of the four spheres, what is this new shape? Is it a square, tetrahedron, circle, triangle, rectangle, sphere, rhombus, or rectangular pyramid? Think step by step, and then put your answer in **bold** as a single phrase (for example, **sphere**). If you don't know, guess.

We found this task to be very difficult for LLMs and a good indicator of reasoning capabilities.
This brought the total number of questions to 1000, and all future updates add and remove the same number, to stay at 1000 total questions.

### 2024-06-24

Removed house traversal task. We discovered ambiguity in the parsing of the answers, and upon fixing it, some models achieved 100% on the task. This made it unsuitable for LiveBench, which aims to only include tasks that are very challenging for even the current best LLMs.

### 2024-06-12

Introducing **LiveBench**: a benchmark for LLMs designed with test set contamination and objective evaluation in mind

LiveBench has the following properties:

 - LiveBench is designed to limit potential contamination by releasing new questions monthly, as well as having questions based on recently-released datasets, arXiv papers, news articles, and IMDb movie synopses.
 - Each question has verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically, without the use of an LLM judge.
 - LiveBench currently contains a set of 17 diverse tasks across 6 categories, and we will release new, harder tasks over time.

Today, we release the initial batch of 960 questions and plan to release sets of questions every month. By doing this, we aim for LiveBench to be free from contamination, since every release will have novel questions.


In addition to contamination, LiveBench avoids the pitfalls of LLM judging by including only questions that have an objective answer. While LLM judging and crowdsourced prompts and evaluations have many benefits, they also introduce significant biases and completely break down when judging the answers to hard questions. For example, we show in our paper that for challenging reasoning and math problems, the pass/fail judgments from GPT-4-Turbo have an error rate of up to 46%.

![changelog-img](assets/livebench-2024-09-30.png)

LiveBench currently evaluates several prominent closed-source models and dozens of open-weight models ranging from 0.5B to 70B in size. LiveBench questions are difficult; for example, GPT-4-Turbo achieves around 50% accuracy on LiveBench overall. Furthermore, in our monthly updates, we will release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future.

We release all questions, code, and model answers, and we welcome community engagement and collaboration for expanding the benchmark tasks and models.
