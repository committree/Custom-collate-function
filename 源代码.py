import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class VarLenDataset(Dataset):
    def __init__(self):
       self.sample = [
        list(range(3)),
        list(range(5)),
        list(range(2)),
        list(range(7)),
       ]

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return torch.tensor(self.sample[idx], dtype=torch.long)

def collate_fn(batch):
        max_len = max(x.size(0) for x in batch)
        padded = []
        for x in batch:
            pad_len = max_len - x.size(0)
            padded.append(torch.nn.functional.pad(x, (0, pad_len), value=0))

        return torch.stack(padded)

def collate_fn_official(batch):
        return pad_sequence(batch, batch_first=True, padding_value=0)

def demo_default_collate():
        print("=" * 60)
        print("[1] 默认 collate（不加 collate_fn）会报错：")
        ds = VarLenDataset()
        dl = DataLoader(ds, batch_size=4)
        try:
            for batch in dl:
                print(batch.shape)
        except RuntimeError as e:
            print("RuntimeError:", e)
            print("→ 原因: default_collate 内部调 torch.stack，要求形状一致")

def demo_custom_collate():
    print("=" * 60)
    print("[2] 手工 collate_fn：")
    ds = VarLenDataset()
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=True)
    for batch in dl:
        print("batch shape:", batch.shape)
        print(batch)

def demo_official_collate():
        print("=" * 60)
        print("[3] 官方 pad_sequence（等价实现）：")
        ds = VarLenDataset()
        dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn_official)
        for batch in dl:
            print("batch shape:", batch.shape)
            print(batch)

if __name__ == "__main__":
    demo_default_collate()
    demo_custom_collate()
    demo_official_collate()
