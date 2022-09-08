import json
import random
import zipfile
from io import BytesIO
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from transformers import BertTokenizer
import os
import sys
from PIL import Image

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))



class SampleDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 data_path: str,
                 test_mode: bool = False):
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.data = []
        for line in lines:
            self.data.append(json.loads(line))

        # 多线程get_visual_feats
        self.handles = [None for _ in range(args.num_workers)]
        # self.handles_pseudo = [None for _ in range(args.num_workers)]

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_dir, 'vocab.txt'), use_fast=True)

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_text(self, text: str, seq_length: int) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=seq_length, padding='max_length', truncation=True)
        # encoded_inputs = self.tokenizer.encode_plus(text, max_length=seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'], dtype=np.int64)
        mask = np.array(encoded_inputs['attention_mask'], dtype=np.int64)
        token_type_ids = np.array(encoded_inputs['token_type_ids'], dtype=np.int64)

        return input_ids, mask, token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load title tokens
        item = self.data[idx]
        title, assignee, abstract = item.get('title', ''), item.get('assignee', ''), item.get('abstract', '')

        text = title + " " + assignee + " " + abstract

        text_input, text_mask, token_type_ids = self.tokenize_text(text, self.bert_seq_length)
        # ocr_input, ocr_mask, ocr_token_type_ids = self.tokenize_text(ocr_text, 128)
        # Step 2, summarize into a dictionary
        data = dict(
            text_input=text_input,
            text_mask=text_mask,
            token_type_ids=token_type_ids,
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = item['label_id']
            data['label'] = torch.LongTensor([label])
        return data
