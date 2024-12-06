## Changelog

### 2024-11-25
This update focused on refreshing questions to check for contamination and increasing the difficulty of tasks for which o1 (and other reasoning models) achieved very high scores.
 - Refreshed the instruction following tasks with new articles from The Guardian
 - Updated IF question generation to include 3 instructions per task on average (previously 2)
 - Refreshed Connections task with new puzzles from NYT
 - Updated Connections generation to more frequently ask for more groups
 - Regenerated Zebra Puzzles, skewing towards larger board sizes and more complex constraints
 - Updated Connections and Zebra Puzzles questions to require answers to be in `<solution><\solution>` tags rather than bolded

### 2024-08-31

This update made no changes to the tasks themselves; instead, it focused on refreshing question sources. A portion of the questions for AMPS_Hard, math_comp, and olympiad were replaced with more recent questions from the respective math competitions.

### 2024-07-26

This update included new questions for the coding completion and LCB generation tasks. It also introduced a new spatial reasoning task that judges models' ability to reason about 2D and 3D shapes. Examples of questions from this task can be seen below:

 > Suppose I have a physical, solid square with vertices ABCD, and I make two cuts through AC and BD. Of the resulting pieces, how many triangles are there?  Think step by step, and then put your answer in **bold** as a single integer (for example, **0**). If you don't know, guess.

 > Suppose I have three solid spheres of radius 5 resting on a plane. Each sphere is tangent to the other two spheres. Now I add a fourth solid sphere of radius 6 in such a way that maximizes the number of tangent points among all pairs of spheres. If I consider a new shape whose vertices are exactly the centers of the four spheres, what is this new shape? Is it a square, tetrahedron, circle, triangle, rectangle, sphere, rhombus, or rectangular pyramid? Think step by step, and then put your answer in **bold** as a single phrase (for example, **sphere**). If you don't know, guess.

We found this task to be very difficult for LLMs and a good indicator of reasoning capabilities.

### 2024-06-31

Removed house traversal task.

### 2024-06-24

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