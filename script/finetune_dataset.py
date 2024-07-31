import numpy as np
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1280, prompt_max_length=256, answer_max_length=1024):
        super().__init__()
        self.dataframe = dataframe
        self.max_length = max_length
        self.prompt_max_length = prompt_max_length
        self.answer_max_length = answer_max_length
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.special_tokens['<bos>']
        self.eos_token = self.tokenizer.special_tokens['<eos>']
        self.pad_token = self.tokenizer.special_tokens.get('<pad>', 0)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        sample = self.dataframe.iloc[index]
        prompt_tokens = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        answer_tokens = self.tokenizer.encode(sample['answer'], add_special_tokens=False)

        prompt_tokens = prompt_tokens[:self.prompt_max_length - 2] if len(prompt_tokens) > self.prompt_max_length else prompt_tokens
        answer_tokens = answer_tokens[:self.answer_max_length - 2] if len(answer_tokens) > self.answer_max_length else answer_tokens

        input_ids = prompt_tokens + [self.bos_token] + answer_tokens + [self.eos_token]
        context_length = len(prompt_tokens) + 1
        padding_length = self.max_length - len(input_ids)
        
        input_ids += [self.pad_token] * padding_length

        loss_mask = [0] * context_length + [1] * (len(input_ids) - context_length - padding_length) + [0] * padding_length

        input_ids = np.array(input_ids, dtype=np.int64)
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = np.array(loss_mask[:-1], dtype=np.int64)
        
        return torch.tensor(X), torch.tensor(Y), torch.tensor(loss_mask)
