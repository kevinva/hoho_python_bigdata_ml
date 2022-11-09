import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import Dataset, DataLoader



class QuoraDataset(Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    def __len__(self):
        return len(self.labels)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(optimizer, scheduler, epoch, model):
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_dataloader)

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        # print(f'model output: {outputs}')
        
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        iter_num += 1
        if iter_num % 100 == 0:
            print(f'epoch: {epoch}, iter_num: {iter_num}, loss: {loss:.4f}, {iter_num / total_iter * 100:.2f}%')
    print(f'Epoch: {epoch}. Average train loss: {total_train_loss / total_iter:.4f}')


def validation(model):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        
        loss = outputs[0]
        logits = outputs[1]
        
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print(f'Accuracy: {avg_val_accuracy:.4f}')
    print(f'Average testting loss: {total_eval_loss / len(test_dataloader):.4f}')
    print('-----------------------------------------------------')


if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv')
    print(train_df.head())

    train_df = train_df[train_df['question2'].apply(lambda x: isinstance(x, str))]
    train_df = train_df[train_df['question1'].apply(lambda x: isinstance(x, str))]

    q1_train, q1_val, q2_train, q2_val, train_label, test_label = train_test_split(
        train_df['question1'].iloc[:],
        train_df['question2'].iloc[:],
        train_df['is_duplicate'].iloc[:],
        test_size = 0.2,
        stratify = train_df['is_duplicate'].iloc[:]
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encoding = tokenizer(list(q1_train), list(q2_train), truncation = True, padding = True, max_length = 100)
    test_encoding = tokenizer(list(q1_val), list(q2_val), truncation = True, padding = True, max_length = 100)

    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    lr = 1e-3
    batch_size = 32
    epoches = 10

    train_dataset = QuoraDataset(train_encoding, list(train_label))
    test_dataset = QuoraDataset(test_encoding, list(test_label))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


    len_dataset = len(train_dataset)

    # 每个epoch有多少个step
    total_steps =  (len_dataset // batch_size) * epoches if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoches

    # 要预热的steps
    warm_up_ratio = 0.1

    optimizer = AdamW(model.parameters(), lr = lr, correct_bias = False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)


    for epoch in range(epoches):
        print(f'-------------------- Epoch: {epoch} --------------------')
        train(optimizer, scheduler, epoch, model)
        validation(model)
