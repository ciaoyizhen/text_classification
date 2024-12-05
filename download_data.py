# -*- encoding: utf-8 -*-
# @Time    :   2024/12/03 16:01:33
# @File    :   download_data.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   下载数据


import os
from tqdm.auto import tqdm
from datasets import load_dataset


def loadConfig(dataset_name:str, split_name:str="train"):
    dataset = load_dataset(dataset_name, split=split_name)
    names = dataset.features["label"].names \
                        if hasattr(dataset.features["label"], "names") \
                        else [] # 这里可能某些数据集不提供names 请手动修改
    with open("configs/label_config.txt", "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

def loadDataset(dataset_name:str, split:str, save_dir:str="data"):
    save_dir = os.path.join(save_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    process_bar = tqdm(dynamic_ncols=True,total=None)
    try:
        iter_data = load_dataset(dataset_name, split=split, streaming=True)
    except:
        print(f"{dataset_name}没有{split}的切割", flush=True)
        return
    for row in iter_data:
        with open(f"{os.path.join(save_dir, split+'.txt')}", "a+", encoding="utf-8")as writer:
            text = row["text"]
            label = row["label"]
            writer.write(f"{text}\t{label}\n")
        process_bar.update(1)
    
if __name__ == "__main__":
    dataset_name = "agentlans/chinese-classification"
    
    loadDataset(dataset_name, "train")
    loadDataset(dataset_name, "validation")
    loadDataset(dataset_name, "test")
    
    loadConfig(dataset_name)
    