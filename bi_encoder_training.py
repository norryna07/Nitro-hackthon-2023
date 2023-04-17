import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

seed = 27

import numpy as np
np.random.seed(seed)
np.random.RandomState(seed)
import random
random.seed(seed)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

trainpath = './train_data.csv'
data = pd.read_csv(trainpath)

#deleted = 0
#i = 0
#while i < data["Text"].shape[0] - deleted:
#    if data["Text"][i].startswith("keyword"):
#        deleted += 1
#        data = data.drop(i)


# Split the dataset into training and testing sets
train_data = data.sample(frac = 0.8, random_state = np.random.RandomState(seed))
test_data = data.drop(train_data.index)
#print(train_data.shape)
#test_data = data.drop(train_data.index)

# Encode the categories as integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['Final Labels'])
test_labels = label_encoder.transform(test_data['Final Labels'])
#print(label_encoder.classes_)

# Load the pre-trained bi-encoder model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))

# Tokenize the text data
train_encodings = tokenizer(list(train_data['Text']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data['Text']), truncation=True, padding=True)


train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_masks = torch.tensor(train_encodings['attention_mask'])
test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_masks = torch.tensor(test_encodings['attention_mask'])
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Create PyTorch dataloaders
batch_size = 4
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


num_epochs = 3
learning_rate = 2e-5
warmup_steps = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Train the model

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 20)
    model.train()
    total_loss = 0
    count = 0
    for batch in train_dataloader:
        count += 1
        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)[0]
        labels = labels.type(torch.LongTensor)
        labels = torch.tensor(labels, device=device)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.cpu().item()
        if count % 50 == 0: 
            print(count)
        del outputs, input_ids, attention_masks, labels, loss
    # Evaluate the model after each epoch
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs[0]
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())
    # Print the classification report for the current epoch
    print(true_labels, predictions)
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs[0]
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())
    # Print the classification report for the current epoch
    print(true_labels, predictions)
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

torch.save(model.state_dict(), 'model.pth')