import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 假设我们有一个自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,json_file,image_path):
        self.json_file=json_file
        self.image_path=image_path
    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx):
        item=self.json_file[idx]
        for i in range(len(item['image'])):
        # Update the image path
            item['image'][i] = self.image_path + item['image'][i]
        return item
    
def collate_fn(item_list):
    return item_list

def create_data_loader(json_file,image_path, batch_size=4, num_workers=4):
    dataset = CustomDataset(json_file,image_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader
