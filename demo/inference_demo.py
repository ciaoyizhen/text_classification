# -*- encoding: utf-8 -*-
# @Time    :   2024/12/05 18:20:15
# @File    :   infer_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   单行推理代码

import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipeline("text-classification", model="output/chinese-classification/checkpoint-49766", device=device)



result = pipe(
    [
        "瞓到差唔多五點幾，無啦啦起咗身瞓唔返",
        "睡得差不多五点多，没啦啦起了身睡不回来",
        "睡得差不多五點多，沒啦啦起了身睡不回來"
    ]
)

print(result)