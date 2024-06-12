# LiveBench - Instruction Following
If we want to do updates on this reach out to Neel Jain <njain17@umd.edu> for the updating the instrucions. Regarding the scoring, reach our to Khalid Saifullah <khalids@umd.edu>. However, you can just each out to either of us.

## Requirements

`pip install -r requirements.txt`

Note you might need to run in a python shell
```
import nltk
nltk.download('punkt')
```

## Generating new data:
Try the following line:

`python live_data.py --total_articles 200 --total_sentences 10`

This will generate 200 unique articles corresponding to a sampled 200 set of instruction.

Some notes: the guardian can only pull 200 articles at once, so for now this works four categories, but if we start to build up, then we might need to pull from two separate days. This will require some coding. However, there are some more todos. For example, currently, it hard coded to assume that we have the four categories that we want. 


## Usage:
### Model Answer Generation:
For OSS model:
```
python LiveBench/livebench/gen_model_answer.py --model-path meta-llama/llama_2_chat_7b_hf --model-id llama_2_chat_7b --dtype bfloat16 --bench-name live_bench/instruction_following
```
Or API model:
```
python LiveBench/livebench/gen_api_answer.py --model gpt-4o-2024-05-13  --bench-name live_bench/instruction_following --max-token 4096
```
### Model Answer Evaluation:
```
python3 -m instruction_following_eval.evaluation_main \
  --input_data=/cmlscratch/khalids/LiveBench/livebench/data/live_bench/instruction_following/question.jsonl \
  --input_response_data=/cmlscratch/khalids/LiveBench/livebench/data/live_bench/instruction_following/model_answer/gpt-4o-2024-05-13.jsonl \
  --output_dir=./instruction_following_eval/data/
```

## Goal

The goal of these tests is to test the instruction following abilities of the instruction tuned language model.

## Current Plan

The current plan is to 

1. Incorporate [IF Eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
2. Add additional tasks from what Tom has and think of other tasks


## TODOs

1. Figure out what is needed to incorporate IF EVAL

### Notes

Path 1:
- filter the current set of prompts to ones that contain numbers and filter.
- The live will be from numbers being generated

Path 2:
- use an LLM based to generate a new example based on the list of transformations and topics.
- Here, we have a choice to either to test just one of the instruction_id_list or multiple at the same time. Hardness would come from the number of composition of the `instruction_id_list`.

Path 3:
- Just do what IF-Eval did, but not the last part.

> First, we generate a set of base prompts with one to three randomly selected verifiable instructions appended to the end of each prompt. Then, we use few-shot prompting to identify illogical prompts and remove them. As the third step, we apply another few-shot prompting based approach to rephrase each prompt, to increase the diversity of phrasing. Finally, we manually check and edit the rephrased prompts one by one.

Neel -- I think we want to create our own prompts using their `instruction_id_list`

This will allow us to have more control of the types of prompts we might want.


### GPT-4o Performance (On IFeval):
```
./instruction_following_eval/data/IFeval/eval_results_strict.jsonl Accuracy Scores:                                                                    
prompt-level: 0.8096118299445472                                                                                                                       
instruction-level: 0.8633093525179856                                                                                                                  
                                                                                                                                                       
change_case 0.8651685393258427                                                                                                                         
combination 0.8307692307692308                                                                                                                         
detectable_content 0.9622641509433962                                                                                                                  
detectable_format 0.910828025477707                                                                                                                    
keywords 0.8098159509202454                                                                                                                            
language 1.0                                                                                                                                           
length_constraints 0.7342657342657343                                                                                                                  
punctuation 0.9393939393939394                                                                                                                         
startend 0.9701492537313433                                                                                                                             2:47 31-May-24

change_case:capital_word_frequency 0.84
change_case:english_capital 0.84
change_case:english_lowercase 0.8974358974358975
combination:repeat_prompt 0.7560975609756098
combination:two_responses 0.9583333333333334
detectable_content:number_placeholders 0.9259259259259259
detectable_content:postscript 1.0
detectable_format:constrained_response 1.0
detectable_format:json_format 0.8823529411764706
detectable_format:multiple_sections 0.9285714285714286
detectable_format:number_bullet_lists 0.7419354838709677
detectable_format:number_highlighted_sections 0.9375
detectable_format:title 1.0
keywords:existence 0.9230769230769231
keywords:forbidden_words 0.8775510204081632
keywords:frequency 0.8809523809523809
keywords:letter_frequency 0.48484848484848486
language:response_language 1.0
length_constraints:nth_paragraph_first_word 0.8333333333333334
length_constraints:number_paragraphs 0.7777777777777778
length_constraints:number_sentences 0.6730769230769231
length_constraints:number_words 0.75
punctuation:no_comma 0.9393939393939394
startend:end_checker 0.9230769230769231
startend:quotation 1.0
```


### GPT-4o Performance (On paraphrase):
```
prompt-level: 0.62
instruction-level: 0.7899159663865546

change_case 0.7692307692307693
combination 0.75
detectable_content 0.6923076923076923
detectable_format 1.0
keywords 0.7333333333333333
length_constraints 0.6666666666666666
punctuation 0.75
startend 1.0

change_case:capital_word_frequency 0.6666666666666666
change_case:english_capital 1.0
change_case:english_lowercase 0.75
combination:repeat_prompt 0.0
combination:two_responses 1.0
detectable_content:number_placeholders 0.42857142857142855
detectable_content:postscript 1.0
detectable_format:constrained_response 1.0
detectable_format:json_format 1.0
detectable_format:multiple_sections 1.0
detectable_format:number_bullet_lists 1.0
detectable_format:number_highlighted_sections 1.0
detectable_format:title 1.0
keywords:existence 0.7142857142857143
keywords:forbidden_words 1.0
keywords:frequency 0.8888888888888888
keywords:letter_frequency 0.2857142857142857
length_constraints:nth_paragraph_first_word 0.5714285714285714
length_constraints:number_paragraphs 0.0
length_constraints:number_sentences 0.625
length_constraints:number_words 0.875
punctuation:no_comma 0.75
startend:end_checker 1.0
startend:quotation 1.0
```

### GPT-4o Performance (On simpler):
```
prompt-level: 0.68
instruction-level: 0.8732394366197183

change_case 0.9411764705882353
combination 0.8333333333333334
detectable_content 0.7857142857142857
detectable_format 1.0
keywords 0.8260869565217391
length_constraints 0.75
punctuation 0.8666666666666667
startend 0.9090909090909091

change_case:capital_word_frequency 1.0
change_case:english_capital 0.8333333333333334
change_case:english_lowercase 1.0
combination:repeat_prompt 0.0
combination:two_responses 1.0
detectable_content:number_placeholders 0.4
detectable_content:postscript 1.0
detectable_format:constrained_response 1.0
detectable_format:json_format 1.0
detectable_format:multiple_sections 1.0
detectable_format:number_bullet_lists 1.0
detectable_format:number_highlighted_sections 1.0
detectable_format:title 1.0
keywords:existence 0.5
keywords:forbidden_words 1.0
keywords:frequency 0.9
keywords:letter_frequency 1.0
length_constraints:nth_paragraph_first_word 0.6666666666666666
length_constraints:number_paragraphs 0.8
length_constraints:number_sentences 0.6666666666666666
length_constraints:number_words 0.8
punctuation:no_comma 0.8666666666666667
startend:end_checker 1.0
startend:quotation 0.75
```

### GPT-4o Performance (On story_generation):
```
prompt-level: 0.7
instruction-level: 0.8604651162790697

change_case 0.9
combination 0.5
detectable_content 0.6666666666666666
detectable_format 0.96875
keywords 0.8461538461538461
length_constraints 0.6842105263157895
punctuation 1.0
startend 1.0

change_case:capital_word_frequency 0.8333333333333334
change_case:english_capital 1.0
change_case:english_lowercase 0.8571428571428571
combination:repeat_prompt 0.0
combination:two_responses 0.6666666666666666
detectable_content:number_placeholders 0.0
detectable_content:postscript 1.0
detectable_format:constrained_response 1.0
detectable_format:json_format 1.0
detectable_format:multiple_sections 0.6666666666666666
detectable_format:number_bullet_lists 1.0
detectable_format:number_highlighted_sections 1.0
detectable_format:title 1.0
keywords:existence 0.9
keywords:forbidden_words 0.8571428571428571
keywords:frequency 1.0
keywords:letter_frequency 0.5
length_constraints:nth_paragraph_first_word 0.6666666666666666
length_constraints:number_paragraphs 0.5
length_constraints:number_sentences 1.0
length_constraints:number_words 0.75
punctuation:no_comma 1.0
startend:end_checker 1.0
startend:quotation 1.0
```

### GPT-4o Performance (On summarize):
```
prompt-level: 0.62
instruction-level: 0.8070175438596491

change_case 0.6666666666666666
combination 0.8571428571428571
detectable_content 0.6153846153846154
detectable_format 1.0
keywords 0.7391304347826086
length_constraints 0.7058823529411765
punctuation 1.0
startend 1.0

change_case:capital_word_frequency 0.4
change_case:english_capital 0.6666666666666666
change_case:english_lowercase 0.8571428571428571
combination:repeat_prompt 0.0
combination:two_responses 1.0
detectable_content:number_placeholders 0.375
detectable_content:postscript 1.0
detectable_format:json_format 1.0
detectable_format:multiple_sections 1.0
detectable_format:number_bullet_lists 1.0
detectable_format:number_highlighted_sections 1.0
detectable_format:title 1.0
keywords:existence 0.2857142857142857
keywords:forbidden_words 1.0
keywords:frequency 1.0
keywords:letter_frequency 0.75
length_constraints:nth_paragraph_first_word 0.3333333333333333
length_constraints:number_paragraphs 0.0
length_constraints:number_sentences 0.8333333333333334
length_constraints:number_words 0.8571428571428571
punctuation:no_comma 1.0
startend:end_checker 1.0
startend:quotation 1.0
```