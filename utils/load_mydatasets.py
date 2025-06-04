import json
import torch
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class zh_cls_fudan_news(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        category = item['category']
        return text, category

if __name__ == '__main__':
    # 加载训练数据
    train_dataset = zh_cls_fudan_news('path/to/train.jsonl')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 加载测试数据
    test_dataset = zh_cls_fudan_news('path/to/test.jsonl')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 打印一些样本数据
    for text, category in train_loader:
        print(f"Text: {text}\nCategory: {category}\n")
        break  # 仅查看一个批次的数据样本
