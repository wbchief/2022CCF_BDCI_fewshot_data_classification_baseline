# encoding=utf-8
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel



base_dir = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


class SampleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_type = args.model_type

        # self.bert = BertModel.from_pretrained(args.bert_dir)
        # bert_output_size = 768
        bert_config = BertConfig.from_pretrained(os.path.join(args.bert_dir, 'config.json'))
        bert_weight = torch.load(os.path.join(args.bert_dir, 'pytorch_model.bin'))
        bert_state_dict = {}
        print(args.bert_dir)
        for k, v in bert_weight.items():
            bert_state_dict[k.replace('bert.', '')] = v
        self.bert = BertModel(config=bert_config)
        msg = self.bert.load_state_dict(bert_state_dict, strict=False)
        print(msg)

        bert_output_size = bert_config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        # 总体分类器
        self.classifier = nn.Linear(bert_output_size, 36)

    def forward(self, text_input, text_mask, token_type_ids):

        sequence_output, pooled_output = self.bert(
            text_input, text_mask, token_type_ids=token_type_ids, return_dict=False
        )

        sequence_output = torch.mean(sequence_output, dim=1)

        sequence_output = self.dropout(sequence_output)

        prediction = self.classifier(sequence_output)
        if self.training:
            return prediction
        else:
            return prediction


if __name__ == '__main__':
    pass
