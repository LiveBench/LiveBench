

declare -a api_models=("gpt-4-turbo-2024-04-09"
"gpt-4-1106-preview"
"gpt-4-0125-preview"
"gpt-3.5-turbo-0125"
"gpt-3.5-turbo-1106"
"gpt-4o-2024-05-13"
"claude-3-opus-20240229"
"claude-3-sonnet-20240229"
"claude-3-haiku-20240307"
"mistral-large-2402"
"mistral-small-2402"
"open-mixtral-8x7b"
"open-mixtral-8x22b"
"command-r"
"command-r-plus"
"gemini-1.5-pro-latest"
)

declare -a oss_models=(
# "mistralai/Mistral-7B-Instruct-v0.2"
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "microsoft/Phi-3-mini-128k-instruct"
# "microsoft/Phi-3-mini-4k-instruct"
# "meta-llama/Llama-2-7b-chat-hf"
# "Qwen/Qwen1.5-0.5B-Chat"
# "Qwen/Qwen1.5-1.8B-Chat"
# "Qwen/Qwen1.5-4B-Chat"
# "Qwen/Qwen1.5-7B-Chat"
# "Nexusflow/Starling-LM-7B-beta"
# "01-ai/Yi-6B-Chat"
# "HuggingFaceH4/zephyr-7b-beta"
# "HuggingFaceH4/zephyr-7b-alpha"
# "lmsys/vicuna-7b-v1.5"
# "lmsys/vicuna-7b-v1.5-16k"
# "microsoft/Phi-3-small-8k-instruct"
# "microsoft/Phi-3-small-128k-instruct"
# "Qwen/Qwen2-0.5B-Instruct"
# "Qwen/Qwen2-1.5B-Instruct"
# "Qwen/Qwen2-7B-Instruct"
# "microsoft/Phi-3-medium-4k-instruct"
# "microsoft/Phi-3-medium-128k-instruct"
"teknium/OpenHermes-2.5-Mistral-7B"
)


# for f in data/live_bench/AMPSmall/*/* 
# do 
#   for model in "${api_models[@]}"
#   do
#     echo  python gen_api_answer.py --model $model  --bench-name ${f#data/} --question-end 10 >> scripts/amps.sh
#   done
# done

# for f in data/live_bench/AMPSmall/*/* 
# do 
#   for model in "${oss_models[@]}"
#   do
#     echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name ${f#data/} --question-end 10 >> scripts/amps.sh
#   done
# done


# for f in data/live_bench/AMPSmall/*/* 
# do 
#   echo  python gen_ground_truth_judgment.py --bench-name ${f#data/} --question-end 10 >> scripts/amps.sh
# done

for model in "${oss_models[@]}"
do
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/instruction_following/paraphrase >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/instruction_following/simplify >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/instruction_following/story_generation >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/instruction_following/summarize >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/coding/LCB_generation >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/coding/coding_completion >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/data_analysis/cta >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/data_analysis/tablejoin >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/data_analysis/tablereformat >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/language/connections >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/language/plot_unscrambling >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/language/typos >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/math/AMPS_hard >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/math/math_comp >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/math/olympiad >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/reasoning/house_traversal >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/reasoning/web_of_lies_v2 >> scripts/gen_livebench_answers.sh
  echo  python gen_model_answer.py --model-path $model --model-id ${model##*/} --dtype bfloat16  --bench-name live_bench/reasoning/zebra_puzzle >> scripts/gen_livebench_answers.sh
done


# for model in "${api_models[@]}"
# do
#   echo  python gen_api_answer.py --model $model --bench-name live_bench/instruction_following/paraphrase >> scripts/gen_livebench_answers.sh
#   echo  python gen_api_answer.py --model $model --bench-name live_bench/instruction_following/simplify >> scripts/gen_livebench_answers.sh
#   echo  python gen_api_answer.py --model $model --bench-name live_bench/instruction_following/story_generation >> scripts/gen_livebench_answers.sh
#   echo  python gen_api_answer.py --model $model --bench-name live_bench/instruction_following/summarize >> scripts/gen_livebench_answers.sh
#   echo  python gen_api_answer.py --model $model --bench-name live_bench/writing/typos >> scripts/gen_livebench_answers.sh
# done
