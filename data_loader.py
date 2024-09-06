from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, input_ids1, attention_masks1, labels1, input_ids2, attention_masks2, labels2):
        self.input_ids1 = input_ids1
        self.attention_mask1 = attention_masks1
        self.labels1 = labels1
        self.input_ids2 = input_ids2
        self.attention_mask2 = attention_masks2
        self.labels2 = labels2

    def __len__(self):
        return len(self.input_ids1)

    def __getitem__(self, idx):
        input_id1 = self.input_ids1[idx]
        am1 = self.attention_mask1[idx]
        label1 = self.labels1[idx]
        input_id2 = self.input_ids2[idx]
        am2 = self.attention_mask2[idx]
        label2 = self.labels2[idx]
        return torch.tensor(input_id1), torch.tensor(am1), torch.tensor(label1), torch.tensor(input_id2), torch.tensor(am2), torch.tensor(label2)

def create_multimodal_dataloader(inputs1, masks1, labels1, inputs2, masks2, labels2, shuffle, batch_size):
    dataset = CustomDataset(inputs1, masks1, labels1, inputs2, masks2, labels2)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader




class URL_CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        am = self.attention_mask[idx]
        label = self.labels[idx]
        return torch.tensor(input_id), torch.tensor(am), torch.tensor(label)

def create_url_dataloader(inputs, masks, labels, shuffle, batch_size):
    dataset = URL_CustomDataset(inputs, masks, labels)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader



class Content_CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        am = self.attention_mask[idx]
        label = self.labels[idx]
        return torch.tensor(input_id), torch.tensor(am), torch.tensor(label)

def create_content_dataloader(inputs, masks, labels, shuffle, batch_size):
    dataset = Content_CustomDataset(inputs, masks, labels)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader