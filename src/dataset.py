# -*- encoding: utf-8 -*-
# @Time    :   2024/08/18 16:20:17
# @File    :   dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   data

import os
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from .utils import getClass

class Dataset():
    def __init__(self, cfg, id2label, label2id, processor) -> None:
        """init

        Args:
            cfg (dict): args of **_dataset in yaml
            id2label (dict):
            label2id (dict):
            processor (Processor): transformers.AutoTokenizer
        """
        
        self.cfg = cfg
        self.id2label = id2label
        self.label2id = label2id
        self.processor = processor
        self.format_map = {
            ".txt": "text",
            ".csv": "csv",
            ".json": "json",
            ".tsv": "csv"
        }
        self._loadData()  # str format not Tensor
        self._process()

    def _chooseFileFormat(self, file_path:str) -> str:
        """get function `load_dataset` format based on file_path

        Args:
            file_path (str): input_file_path
        
        Returns:
            str: function `load_dataset` format

        Example:
            >>> format_ = self._chooseFileFormat("data/file.txt")
            >>> print(format_)
            "text"
        """
        _, ext = os.path.splitext(file_path)
        format_ = self.format_map.get(ext, None)
        assert format_ is not None, f"file_path only support {set(self.format_map.keys())}"
        if format_ not in {"text"}:
            raise NotImplementedError(f"currently not implement {ext}")
        return format_

    def _loadData(self):
        print("loading Data...")
        data_path_list = self.cfg["data_paths"]
        assert len(data_path_list) != 0, "data_paths length not be zero!"
        
        if len(data_path_list) == 1:
            data_path = data_path_list[0]
            format_ = self._chooseFileFormat(data_path)
            self.data = load_dataset(format_, data_files=data_path, split="train")
        else:        
            datasets = []
            for data_path in data_path_list:
                format_ = self._chooseFileFormat(data_path)
                datasets.append(load_dataset(format_, data_files=data_path, split="train"))
            self.data = concatenate_datasets(datasets)



    def _process(self):
        
        
        def process(example):
            lines = example["text"]
            text_list = []
            label_list = []
            for line in lines:
                if line := line.strip():
                    text, label = line.split("\t")
                    text_list.append(text)
                    label = self.label2id[label]
                    label_list.append(label)
            return {
                "text": text_list,
                "label": label_list
            }
                     
            
        def tokenizer(example):
            return self.processor(example["text"], truncation=True)
        
        self.data = self.data.map(process, batched=True)
        self.data = self.data.map(tokenizer, batched=True)