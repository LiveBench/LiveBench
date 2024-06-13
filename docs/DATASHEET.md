# Datasheet for LiveBench

We include a [Datasheet](https://arxiv.org/abs/1803.09010). 
Thanks for the Markdown template from [Christian Garbin's repository](https://github.com/fau-masters-collected-works-cgarbin/datasheet-for-dataset-template).

Jump to section:

- [Motivation](#motivation)
- [Dataset Composition](#dataset-composition)
- [Collection Process](#collection-process)
- [Data Preprocessing](#data-preprocessing)
- [Data Distribution](#data-distribution)
- [Dataset Maintenance](#dataset-maintenance)
- [Legal and Ethical Considerations](#legal-and-ethical-considerations)

## Motivation

### Why was the datasheet created? (e.g., was there a specific task in mind? was there a specific gap that needed to be filled?)

The goal of releasing LiveBench is to accelerate research in LLMs by introducing a benchmark that avoids the pitfalls of test set contamination and LLM judging. Specfically, many existing benchmarks are either static, and therefore may become obsolete as new LLMs train on their test data, or use humans or LLMs to create prompts or judgments, which introduces errors and biases. LiveBench avoids all of these issues, and it is also diverse, with six different categories: math, reasoning, coding, data analysis, language, and instruction following.

### Has the dataset been used already? If so, where are the results so others can compare (e.g., links to published papers)?

The benchmark is new, and it has only been used by us in an initial evaluation of 34 LLMs. We have released all of the results, including raw model outputs and judgments.

### What (other) tasks could the dataset be used for?

The dataset consists of questions meant for LLMs, and we do not foresee it being very useful for anything other than evaluating ML models. The raw model outputs we released could potentially be used to fine-tune LLMs, for example with DPO. 


### Who funded the creation of the dataset? 

This benchmark was funded by Abacus.AI.

### Any other comments?

None.

## Dataset Composition

### What are the instances?(that is, examples; e.g., documents, images, people, countries) Are there multiple types of instances? (e.g., movies, users, ratings; people, interactions between them; nodes, edges)

Each instance is an English language question. Other than the question, there is the ground truth and other metadata.

### How many instances are there in total (of each type, if appropriate)?

There are 960 total questions. For a breakdown of the number of questions per task, see the our paper.

### What data does each instance consist of ? “Raw” data (e.g., unprocessed text or images)? Features/attributes? Is there a label/target associated with instances? If the instances are related to people, are subpopulations identified (e.g., by age, gender, etc.) and what is their distribution?

The data consists of English language questions. There is a ground truth associated with each instance. The instances are not related to people.

### Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

There is no missing information.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.

There are no relationships between individual instances.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).

For many tasks, the instances are samples generated from a distribution, or they contain data drawn from a set. For example, the coding questions consists of only medium and hard questions from LeetCode (via LiveCodeBench) which also had answers on GitHub. See our paper for the full details of how questions were sampled for each task.

### Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

The benchmark is not meant to be used for training; it is meant for evaluation only. So, all 960 questions are meant as a test set. Although, the benchmark as a whole is also meant to be more immune to unintentional or intentional contamination, by having monthly udates and using recent information sources for questions.

### Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

There are no known errors, sources of noise, or redundancies.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

The dataset is self-contained.

### Any other comments?

None.


## Collection Process


### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?

Each task was created in a different way. Some were created with software alone, such as Web of Lies V2 and House Traversal. Some were created with a mix of software and an API that pulls certain information, such as plot unscrambling or typo correction. In many cases, the software has validation to make sure the ground truth is correct, and we also validated questions ourselves.

### How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.
 
Some of the questions use recent data, such as datasets from Kaggle, abstracts from arXiv, or movie synopses from IMDb. These were collected with their respective APIs.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
 
In most cases, the sampling was probabilitic with certain sampling probabilities. In some cases such as the Connections questions, we used the 50 most recent puzzles.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
 
The creation of LiveBench was done by the authors of this work.

### Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.
 
The timeframe for constructing the initial LiveBench release was from April 1, 2024 to June 12, 2024.
Most questions consist of only the question or only data after April 1, 2024.
There are some tasks that contain information from before April 1, 2024, for example, some IMO questions in the math section are from July 2, 2023. 

## Data Preprocessing

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.

There was no preprocessing done on the questions.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.

There was no preprocessing done. The raw model answers are available at https://huggingface.co/datasets/livebench/model_answer.

### Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.

N/A

### Does this dataset collection/processing procedure achieve the motivation for creating the dataset stated in the first section of this datasheet? If not, what are the limitations?
 
We hope that the release of this benchmark will achieve our goal of accelerating research in LLMs by introducing a benchmark that avoids the pitfalls of test set contamination and LLM judging. Time will tell whether our suite will be adopted by the community.

### Any other comments
 
None.

## Dataset Distribution

### How will the dataset be distributed? (e.g., tarball on website, API, GitHub; does the data have a DOI and is it archived redundantly?)
 
The data is available on HuggingFace at https://huggingface.co/livebench.

### When will the dataset be released/first distributed? What license (if any) is it distributed under?
 
The benchmark suite is public as of June 12, 2024, distributed under the Apache License 2.0.

### Are there any copyrights on the data?
 
There are no copyrights on the data.

### Are there any fees or access/export restrictions?
 
There are no fees or restrictions.

### Any other comments?
 
None.

## Dataset Maintenance

### Who is supporting/hosting/maintaining the dataset?
 
The authors of this work are supporting/hosting/maintaining the dataset.

### Will the dataset be updated? If so, how often and by whom?

The dataset will be updated monthly with new questions, including new tasks and harder versions of the current tasks. We also plan to run new LLMs on the benchmark as new LLMs are released. We also welcome contributions from the community. See our contributing policy for more details.

### How will updates be communicated? (e.g., mailing list, GitHub)
 
Updates will be communicated on the GitHub README: https://github.com/LiveBench/LiveBench.

### If the dataset becomes obsolete how will this be communicated?
 
If the dataset becomes obsolete, it will be communicated on the GitHub README: https://github.com/LiveBench/LiveBench.

### If others want to extend/augment/build on this dataset, is there a mechanism for them to do so? If so, is there a process for tracking/assessing the quality of those contributions. What is the process for communicating/distributing these contributions to users?
 
Others can create a pull request on GitHub with possible extensions to our benchmark, which will be approved case-by-case. These updates will again be communicated on the GitHub README.


## Legal and Ethical Considerations

### Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.
 
There was no ethical review process. Note that our data does not involve human subjects.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals non-public communications)? If so, please provide a description.
 
The datasets do not contain any confidential data.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
 
None of the data might be offensive, insulting, threatening, or otherwise cause anxiety.

### Does the dataset relate to people? If not, you may skip the remaining questions in this section.
 
The datasets do not relate to people.

### Any other comments?
 
None.








