import json
import numpy as np
from nltk_utils import tokenize,extract_stop_words,stem,bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

with open('intents.json',encoding='UTF-8') as f:
    intents = json.load(f)

all_words = list()
tags = list()
pattern_tag_data = list()

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        words = tokenize(pattern)
        words_without_stops = extract_stop_words(words)
        all_words.extend(words_without_stops)
        pattern_tag_data.append((words_without_stops,tag))

ignore_words = ['?','!','.',',',';','#']

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))

tags = sorted(set(tags))

x_train = list()
y_train = list()

for (pattern_sentece,tag) in pattern_tag_data:
    bag = bag_of_words(pattern_sentece,all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).long()
class ChatDataset(Dataset):
    def __init__(self):
        self.number_of_samples = len(x_train)
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self,index):
        return self.x_train[index], self.y_train[index]
    
    def __len__(self):
        return self.number_of_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset, batch_size=8, shuffle=True, num_workers = 0)

input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size,hidden_size,output_size).to(device)

criterion = nn.CrossEntropyLoss()
epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

for epoch in range(epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        output = model(words)
        loss = criterion(output,labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss = {loss.item():.4f}')

print(f'Final loss = {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words" : all_words,
    "tags": tags
}

DATA_FILE = 'data.pth'

torch.save(data,DATA_FILE)

print(f'Training complete. File saved to {DATA_FILE}')