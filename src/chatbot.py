import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import tokenize,extract_stop_words,stem,bag_of_words
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json',encoding='UTF-8') as f:
    intents = json.load(f)

DATA_FILE = 'data.pth'
data = torch.load(DATA_FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "NavlunBot"
os.system('cls')
print("Merhaba. Ben NavlunBot. Sana nasıl yardımcı olabilirim ?")
while True:
    sentence = input('Me : ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    sentence = extract_stop_words(sentence)
    x = bag_of_words(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x)
    out = model(x)
    _,pred = torch.max(out,dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(out, dim=1)
    actual_prob = probs[0][pred.item()]

    if actual_prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name} : {random.choice(intent['responses'])}")
                if(tag == "shipment-payment"):
                    print(f"{bot_name} : 5 sn sonra ilgili sayfaya yönlendirileceksiniz.")
                    time.sleep(5)
                    os.system("start \"\" https://navlungo.com/ship/searchs")
    else:
        print(f"{bot_name} : Buna cevap veremiyorum :(")
