import argparse
import math
import time
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm as tdqm
from transformers import BertTokenizer, BertModel

def read_obqa(file_name):
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    for i in range(5):
        print(i,data[i])
    print('data: %d' % (len(data)))
    return(data)


class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Bertmodel = BertModel.from_pretrained("bert-base-uncased")
        self.layer = nn.Linear(768,1)
    def forward(self, x):
        outputs = self.Bertmodel(**x)
        #grabs embedding for class token inserted at start of seq
        cls = outputs.last_hidden_state[:, 0, :]
        transformed = self.layer(cls)
        return transformed.squeeze(-1)


def Tokenize(opt):
    #precomputing tokenizations
    cnt=0
    opt.train_tokenized = []
    opt.train_labels = []
    opt.valid_tokenized = []
    opt.valid_labels = []
    opt.test_tokenized = []
    opt.test_labels = []

    for data in [opt.train, opt.valid, opt.test]:
        for question in tdqm(data, desc="Precomputing tokenizations"):
            texts = []
            fact = "" if opt.zero_shot else question['fact']
            #iter through options
            for option in ["A", "B", "C", "D"]:
                text = "[CLS] " + fact + " " + question['stem'] + " " + question[option]
                texts.append(text)
            #tokenize
            tokenized = opt.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=128)
            label = np.argmax([1 if x == question['Answer'] else 0 for x in ["A", "B", "C", "D"]])
            if cnt==0:
                opt.train_tokenized.append(tokenized)
                opt.train_labels.append(label)
            elif cnt==1:
                opt.valid_tokenized.append(tokenized)
                opt.valid_labels.append(label)
            elif cnt==2:
                opt.test_tokenized.append(tokenized)
                opt.test_labels.append(label)
        cnt+=1
    

def train(model, opt):
    scaler = torch.amp.GradScaler()
    prevaccuracy = 0
    model = model.to(opt.device)
    #epoch loop
    for epoch in tdqm(range(opt.epochs), desc="epochs"):
        #shuffle data
        ind = np.random.permutation(len(opt.train_tokenized))
        opt.train_tokenized = [opt.train_tokenized[i] for i in ind]
        opt.train_labels = [opt.train_labels[i] for i in ind]
        #batch
        for  idx in tdqm(range(0,len(opt.train_labels),opt.batchsize), desc=f"Batches"):
            #define batch
            batch_examples = opt.train_tokenized[idx: idx + opt.batchsize]
            batch_targets = torch.tensor(opt.train_labels[idx: idx + opt.batchsize]).to(opt.device)
            #flatten for single pass and move to gpu
            questionoptions = {k: torch.cat([question[k] for question in batch_examples], dim=0).to(opt.device) for k in batch_examples[0].keys()}
            with torch.cuda.amp.autocast():
              #toekenize and pass to model
              pred = model(questionoptions)
              pred = pred.view(len(batch_examples),4)
              loss = opt.lossfunc(pred,batch_targets)
            opt.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt.optimizer)
            scaler.update()

        accuracy = test_model(model, opt, validation = True)
        if accuracy < prevaccuracy:
            #Early Stopping based on validation set
            break
        prevaccuracy = accuracy
        print("Accuracy is ", accuracy)
        #save model weights
        os.makedirs("BertQAweights", exist_ok = True)
        intacc = int(accuracy * 100)
        filename = f"BertQAweights/model_weights_zero_shot_{intacc}.pt" if opt.zero_shot else f"BertQAweights/model_weights_{intacc}.pt"
        torch.save(model.state_dict(), filename)
        torch.cuda.empty_cache()


def test_model(model, opt, validation = False):
    examples = opt.valid_tokenized if validation else opt.test_tokenized
    labels = opt.valid_labels if validation else opt.test_labels
    model.eval()
    correct = 0
    with torch.no_grad():
        for  idx in tdqm(range(0,len(examples),opt.batchsize), desc=f"Evaluation on {"test" if not validation else "validation"} set"):
           #define batch
            batch_examples = examples[idx: idx + opt.batchsize]
            batch_targets = torch.tensor(labels[idx: idx + opt.batchsize]).to(opt.device)
            #flatten for single pass and move to gpu
            questionoptions = {k: torch.cat([question[k] for question in batch_examples], dim=0).to(opt.device) for k in batch_examples[0].keys()}
            
            #toekenize and pass to model
            pred = model(questionoptions)
            pred = pred.view(len(batch_examples),4)
            preds = torch.argmax(pred, dim=1)
            correct += sum([1 if pred == target else 0 for pred, target in zip(preds, batch_targets)])
    return correct/len(labels)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=3)
    parser.add_argument("-batchsize", type=int, default=100)
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-lr", type=float, default=0.00001)
    parser.add_argument("-zero_shot", type=int, default=0)
    opt = parser.parse_args()
    opt.zero_shot = False if opt.zero_shot == 0 else True
    
    opt.train = read_obqa('obqa/obqa.train.txt')
    opt.test = read_obqa('obqa/obqa.test.txt')
    opt.valid = read_obqa('obqa/obqa.valid.txt')
    
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
    opt.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    torch.cuda.empty_cache()
    Tokenize(opt)
    model = QAModel()
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    opt.lossfunc = nn.CrossEntropyLoss()
    train(model, opt)

    accuracy = test_model(model, opt)
    print("Test set accuracy is:", accuracy)

if __name__ == "__main__":
    main()