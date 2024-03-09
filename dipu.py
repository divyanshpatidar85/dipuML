import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv('ehealthforumQAs_modified.csv')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


encoded_data = tokenizer(df['question'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['answer'])

# Define a custom dataset
class QADataset(Dataset):
    def __init__(self, encoded_data, labels):
        self.encoded_data = encoded_data
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: self.encoded_data[key][idx] for key in self.encoded_data}, self.labels[idx]

# Create a DataLoader
dataset = QADataset(encoded_data, df['label'].tolist())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch size to 1

# Load the pre-trained GPT-2 model for sequence classification and adjust output layer
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(df['answer'].unique()))
model.resize_token_embeddings(len(tokenizer))

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)


for epoch in range(3):
    for batch in dataloader:
        inputs, labels = batch

        inputs['input_ids'] = inputs['input_ids'][:, :512]
        inputs['attention_mask'] = inputs['attention_mask'][:, :512]

        optimizer.zero_grad()
        
        labels = labels.view(-1)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch} Loss: {loss.item()}')


model.save_pretrained('content')
