# Contributing to LiveBench

Thanks for taking the time to contribute!

## Reporting bugs

To report a bug, please [open an issue](https://github.com/LiveBench/LiveBench/issues) or email the authors.
When you are creating a bug report, please include as many details as possible. 
For example, describe the bug, include a small script to reproduce the bug, and give the expected behavior of the script.

## Code contribution

Contributions are welcome! Feel free to open a pull request to the main branch.

## Model contribution

We'd be happy to add more models to the leaderboard (depending on demand). Please send your model or evals requests to [livebenchai@gmail.com](mailto:livebenchai@gmail.com) in one of these formats 

- Huggingface model, while also telling us the [fastchat model adapter](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py) to use
- API endpoint, ideally using the OpenAI api format (If not, please submit a PR with the necessary code changes to run this model)
- The above methods are preferred, but you can also send instructions on how to run your model if it differs from the above methods.
- If you want to run the model yourself, please evaluate it using the commands in the [README](https://github.com/LiveBench/LiveBench?tab=readme-ov-file#usage), and then send us the model_answer jsonl files. You can simply zip the `data/live_bench` folder that is created when you evaluate your model.
