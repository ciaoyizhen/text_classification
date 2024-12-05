# Text Classification

You can track the latest updates by visiting the project's github address：[Text Classification Repository](https://github.com/ciaoyizhen/text_classification)

## update
```
2024.12.05 first commit
```
## Requirement:

```
python >= 3.10
```

## Goal
Use huggingface to implement a variety of tasks, and you can replace the model at any time without modifying the code.

## Train Step:
```
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. modify yaml config
5. torchrun main.py (yaml_path) or python main.py
```
> Note: multi-gpu  use  torchrun --nproc_per_node=x main.py your_yaml

## Eval Step:
```
python demo/inference_demo.py
```




## FAQ
1. open too many file
```
ulimit -n xxx  # increase open file
```
2. How to download a model to train
```
1. open this (https://huggingface.co/models)
2. choose and download a model
3. modify yaml
```
3. Multi Gpu how to train
```
torchrun --nproc-per-node=x main.py configs/test.yaml

see more in (https://pytorch.org/docs/stable/elastic/run.html)
```



