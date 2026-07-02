import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        # old eager implementation — materialized a tensor per window upfront,
        # which is a slow, single-threaded Python loop on large corpora
        # self.input_ids = []
        # self.target_ids = []
        #
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        #
        # for i in range(0, len(token_ids) - max_length, stride):
        #     input_chunk = token_ids[i:i + max_length]
        #     target_chunk = token_ids[i+1: i + max_length + 1]
        #     self.input_ids.append(torch.tensor(input_chunk))
        #     self.target_ids.append(torch.tensor(target_chunk))

        self.token_ids = torch.tensor(
            tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        )
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        # old: return len(self.input_ids)
        return (len(self.token_ids) - self.max_length) // self.stride

    def __getitem__(self, index) -> tuple:
        # old: return self.input_ids[index], self.target_ids[index]
        i = index * self.stride
        input_chunk = self.token_ids[i:i + self.max_length]
        target_chunk = self.token_ids[i + 1:i + self.max_length + 1]
        return input_chunk, target_chunk


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
