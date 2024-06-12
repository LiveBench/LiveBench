import requests
import json
import numpy as np
import nltk
nltk.download('punkt')  # Download the necessary resources

def get_news_articles(total_articles, api_key):
    total_articles = min(total_articles,200) # only 200 articles can be fetched at a time
    # Get the news articles from the Guardian API
    i = 0
    print("Getting news articles", total_articles)
    body_texts = []
    # These should be live but there is a way to do from DATE in the url search
    url = f'https://content.guardianapis.com/search?page-size={total_articles}&type=article&api-key={api_key}&show-fields=all'
    response = requests.get(url)
    response.raise_for_status()
    news_data = response.json()
    for article in news_data['response']['results']:
        body_text = article['fields']['bodyText']
        body_texts.append(body_text)
        i += 1
    return body_texts

def extract_n_sentences(article_text, total_sentences):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(article_text)
    new_text = ""
    for i, sentence in enumerate(sentences):
        if i+1 > total_sentences:
          break
        new_text += " "+sentence
    return new_text


def get_subtask_prompt(task):
  if task == "paraphrase":
    return "Please paraphrase based on the sentences provided."
  elif task == "summarize":
    return "Please summarize based on the sentences provided."
  elif task == 'simpler':
    return "Please explain in simpler terms what this text means."
  elif task == 'story_generation':
    return "Please generate a story based on the sentences provided."


def add_instructions_to_registry(no_of_instructions_to_add, max_instructions=5):
    ### THIS SEEMS INSANE TO PUT HERE BUT WE MIGHT WANT TO SAMPLE IN A WAY THAT WE CAN CONTROL ###
    length_constraints=["length_constraints:number_sentences", "length_constraints:number_paragraphs", \
        "length_constraints:number_words", "length_constraints:nth_paragraph_first_word"]
    keyword_constraints=["keywords:existence", "keywords:frequency", "keywords:forbidden_words", \
        "keywords:letter_frequency"]
    content_constraints=["detectable_content:number_placeholders", "detectable_content:postscript"]
    format_constraints=["detectable_format:number_bullet_lists", "detectable_format:constrained_response", \
        "detectable_format:number_highlighted_sections", "detectable_format:multiple_sections", \
        "detectable_format:json_format", "detectable_format:title"]
    combinations_constraints=["combination:two_responses","combination:repeat_prompt"]
    startend_constraints=["startend:end_checker", "startend:quotation"]
    changecase_constraints=["change_case:capital_word_frequency","change_case:english_capital", "change_case:english_lowercase"]
    punctuation_constraints=["punctuation:no_comma"]
    
    all_constraints = length_constraints + keyword_constraints + content_constraints + format_constraints + \
        combinations_constraints + startend_constraints + changecase_constraints + punctuation_constraints
    # starting with two because then you will have a bunch that are just one instruction
    samples_to_draw = np.random.randint(2, max_instructions+1, no_of_instructions_to_add)
    instruction_id_list_of_lists = []
    for draw in samples_to_draw:
        instruction_id_list_of_lists.append(np.random.choice(all_constraints, draw, replace=False).tolist())
    return instruction_id_list_of_lists

def disallowed_pairs(conflicts):
    created_disallowed_pairs = []
    for key in conflicts:
        for value in conflicts[key]:
            created_disallowed_pairs.append([key, value])
    return created_disallowed_pairs


def check_for_conflitcs(instruction_lists):
    from instruction_following_eval import instructions_registry
    # construct a list of disallowed pairs
    conflicts_dict = instructions_registry.INSTRUCTION_CONFLICTS
    created_disallowed_pairs = disallowed_pairs(conflicts_dict)
    created_disallowed_pairs_org = created_disallowed_pairs.copy()
    
    deconflicted_instructions = []
    for instruction_set in instruction_lists:
        deconflicting_instruction_ = instruction_set.copy()
        # shuffle so there is no bias
        np.random.shuffle(created_disallowed_pairs)
        for disallowed_pair in created_disallowed_pairs:
            if set(disallowed_pair).issubset(set(deconflicting_instruction_)) and len(set(disallowed_pair)) != 1:
                # it worth noting that certain ones have more conflicts than others. You can construct a heiarchy if wanted.
                # looking set(INSTRUCTION_DICT.keys()).difference in instructions_registry.py
                removed_element = np.random.choice(disallowed_pair)
                deconflicting_instruction_.remove(removed_element)
                # print("Removing ", removed_element, "Conflicted pair", disallowed_pair," From", instruction_set, )
        created_disallowed_pairs = created_disallowed_pairs_org.copy()
        deconflicted_instructions.append(list(set(deconflicting_instruction_)))
    return deconflicted_instructions

def get_kwargs_and_descriptors(instruction):
    # Get the kwargs for the instruction
    from instruction_following_eval import instructions_registry
    build_instruction = instructions_registry.INSTRUCTION_DICT[instruction](instruction)
    instruction_text = build_instruction.build_description()
    kwargs = build_instruction.get_instruction_args()
    return instruction_text, kwargs

def get_all_kwargs_and_descriptors(instruction_lists):
    # Get all kwargs for all instructions
    all_kwargs = []
    instruction_all_texts = []
    for instruction_set in instruction_lists:
        instruction_kwargs = []
        instruction_combined_text = ""
        for instruction in instruction_set:
            instruction_text, kwargs = get_kwargs_and_descriptors(instruction)
            if kwargs is None:
                kwargs = {}
            instruction_kwargs.append(kwargs)
            instruction_combined_text += " "+instruction_text
        assert len(instruction_kwargs) == len(instruction_set)
        all_kwargs.append(instruction_kwargs)
        instruction_all_texts.append(instruction_combined_text)
    return all_kwargs, instruction_all_texts


def get_instruction_kwargs_list(no_of_instructions_to_add, max_instructions=5):
    no_of_instructions_to_add = min(200, no_of_instructions_to_add)
    instruction_lists = add_instructions_to_registry(no_of_instructions_to_add, max_instructions)
    deconflicted_instructions = check_for_conflitcs(instruction_lists)
    all_kwargs, instruction_all_texts = get_all_kwargs_and_descriptors(deconflicted_instructions)
    
    return all_kwargs, instruction_all_texts, deconflicted_instructions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_articles", type=int, default=200)
    parser.add_argument("--total_sentences", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="output.jsonl")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    print(args)

    # Get the news articles and the task prompt
    news_articles = get_news_articles(args.total_articles+10, args.api_key)
    tasks = ["paraphrase", "summarize", simplify, "story_generation"]
    print("Number of articles: ", len(news_articles))
    
    prompt="""The following are the beginning sentences of a news article from the Guardian.\n-------\n{0}\n-------\n{1} {2}"""

    constraints_kwargs_all, constraints_text, constraint_lists = get_instruction_kwargs_list(len(news_articles))
    assert len(constraints_kwargs_all) == len(constraints_text) == len(constraint_lists) == len(news_articles); "Lengths do not match."

    
    from utils import plot_histogram
    plot_histogram([len(x) for x in constraint_lists], "Number of Instructions (IFEval Live Bench)", "Number of Instructions", "Frequency", bins=5)

    root_output_file = args.output_file.split(".")[0]
    out_file_type = args.output_file.split(".")[1]
    ##### 
    task_index = 0
    task = tasks[task_index]
    total_written = 0
    for article, constraints_kwargs, constraint_text, constraint_list_ in zip(news_articles, constraints_kwargs_all, constraints_text, constraint_lists):
        task_prompt = get_subtask_prompt(task)
        full_output_file = root_output_file+"_"+task+"."+out_file_type
        
        extracted_text = extract_n_sentences(article, args.total_sentences)
        if len(extracted_text) == 0:
            continue
            
        generated_prompt = prompt.format(extracted_text, task_prompt, constraint_text)
        if "combination:repeat_prompt" in constraint_list_:
            index = constraint_list_.index("combination:repeat_prompt")
            constraints_kwargs[index]["prompt_to_repeat"] = generated_prompt.split("First repeat the request word for word without change,")[0]
            # print("Repeating prompt: ", constraints_kwargs[index]["prompt_to_repeat"])
            # print("Original Prompt: ", generated_prompt)

            
        with open(full_output_file, "a") as f:
            prompt_json = {"question_id": total_written, "task": task, "turns":[generated_prompt], "category":"instruction_following", "instruction_id_list":constraint_list_, "kwargs":constraints_kwargs, "task_prompt":task_prompt}
            json.dump(prompt_json, f)
            f.write("\n")
        

        
        total_written += 1
        if total_written >= args.total_articles//len(tasks):
            print("Total Written for task: ", task, " is ", total_written)
            print("Saved to file: ", full_output_file, " with task: ", task)
            if args.total_articles == total_written:
                break
            task = tasks[task_index+1]
            task_index+=1
            print("Switching to task: ", task, " with index: ", task_index)
            total_written = 0


    print("Total written: ", total_written)


