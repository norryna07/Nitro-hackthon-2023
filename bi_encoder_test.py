import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Load the pre-trained bi-encoder model and tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

# Load the saved weights
model.load_state_dict(torch.load('model.pth'))

# Prepare the data for inference
batch_size = 4
testpath = './test_data.csv'
data = pd.read_csv(testpath)
text = data['Text']
encodings = tokenizer(list(text), truncation=True, padding=True)
test_input_ids = torch.tensor(encodings['input_ids'])
test_attention_masks = torch.tensor(encodings['attention_mask'])

train_data = pd.read_csv('./train_data.csv')
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['Final Labels'])
ids = torch.tensor(list(data['Id']))

test_data = TensorDataset(test_input_ids, test_attention_masks, ids)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Perform inference
labels = []
with torch.no_grad():
    model.eval()
    for batch in test_dataloader:
        (input_ids, attention_masks, ids) = tuple(t.to(device) for t in batch)
        output = model(input_ids=input_ids, attention_mask=attention_masks)[0]
        predictions = torch.argmax(output, dim=1).tolist()
        labels.extend(np.around(np.array(predictions)))
data['Labels'] = label_encoder.classes_[labels]
data.drop(columns = ['Text']).to_csv('bi_encoder_result.csv', index=False)