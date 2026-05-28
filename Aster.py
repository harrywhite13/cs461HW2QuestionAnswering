import argparse
import math
import time
from tqdm.notebook import tqdm as tdqm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import os

class GPT2Attention(nn.Module):
    def __init__(self, d_model, heads, max_seq_len, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()

        assert d_model % heads == 0

        self.heads = heads
        self.d_model = d_model
        self.head_dim = d_model // heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        bias = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        attn = torch.matmul(probs, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        return self.resid_dropout(self.c_proj(attn))


class GPT2MLP(nn.Module):
    def __init__(self, d_model, d_ff, resid_dropout=0.0):
        super().__init__()

        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)

        try:
            self.act = nn.GELU(approximate="tanh")
        except TypeError:
            self.act = nn.GELU()

        self.dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class GPT2Block(nn.Module):
    def __init__(self, d_model, heads, d_ff, max_seq_len, attn_dropout=0.0, resid_dropout=0.0, layer_norm_epsilon=1e-5):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.attn = GPT2Attention(d_model, heads, max_seq_len, attn_dropout=attn_dropout, resid_dropout=resid_dropout)

        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.mlp = GPT2MLP(d_model, d_ff, resid_dropout=resid_dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, seqlen, d_ff, dropout=0.0, layer_norm_epsilon=1e-5):
        super().__init__()

        self.seqlen = seqlen

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seqlen, d_model)

        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([GPT2Block(d_model, heads, d_ff, seqlen, attn_dropout=dropout, resid_dropout=dropout, layer_norm_epsilon=layer_norm_epsilon) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.size()

        if seq_len > self.seqlen:
            raise ValueError(f"sequence length {seq_len} exceeds model seqlen {self.seqlen}")

        pos = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return x, logits


def load_model_best_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu",weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    return state_dict

def my_tokenizer(path, tokenizer, max_tokens):
    indices = []
    line_batch = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            encoded = tokenizer(line, add_special_tokens=False)["input_ids"]
            if len(indices) < max_tokens:
                for ids in encoded:
                    indices.append(int(ids))
    return indices

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
            tokenized = []
            fact = "" if opt.zero_shot else question['fact']
            text = opt.tokenizer(fact + " " + question['stem'] + " ", add_special_tokens=False)["input_ids"]
            choicestart = len(text)
            #FIRST IN LIST IS LENGTH OF PROMPT
            text.insert(0,choicestart)
            #iter through options
            for option in ["A", "B", "C", "D"]:
                answerchoice = opt.tokenizer(question[option], add_special_tokens=False)["input_ids"]
                tokenized.append(text + answerchoice)
            #tokenize
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

@torch.no_grad()
def test_model(model, indices, opt, epoch=0, device=None):
    start_time = time.time()

    if device is None:
        device = next(model.parameters()).device

    aa = opt.eval_seqlen
    bb = opt.eval_batchsize
    total_loss = 0.0
    count = 0

    n_tokens = len(indices)
    vocab_size = model.wte.weight.size(0)
    stride = aa * bb

    with torch.no_grad():
        for i in range(0, n_tokens - aa + 1, stride):
            src = torch.zeros((bb, aa), dtype=torch.long)
            trg = torch.zeros((bb, aa - 1, vocab_size), dtype=torch.float)
            actual_batchsize = 0
            for k in range(bb):
                start_idx = i + k * aa
                if start_idx + aa > n_tokens:
                    break
                for j in range(aa-1):
                    src[k, j] = indices[start_idx + j]
                    next_token_id = indices[start_idx + j + 1]
                    trg[k, j, indices[start_idx + j + 1]] = 1.0

                actual_batchsize += 1

            src = src[:actual_batchsize].to(device)
            trg = trg[:actual_batchsize].to(device)

            attention_mask = torch.ones_like(src, dtype=torch.long, device=device)

            x,preds = model(src, attention_mask=attention_mask)

            preds = preds[:, :-1, :]

            max_preds = torch.amax(preds, dim=2).unsqueeze(2)
            preds = preds - max_preds
            logits = torch.exp(preds)
            denoms = torch.sum(logits, 2)
            denoms = denoms.unsqueeze(2)
            numer = logits * trg
            numer = torch.sum(numer, 2)
            numer = numer.unsqueeze(2)
            probs = numer / denoms
            loss = -torch.log(probs + 1e-12).mean()
            print(i,loss.item())

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    ppl = math.exp(min(avg_loss, 20.0))

    elapsed_min = int((time.time() - start_time) // 60)

    print(" ")
    print("%dm: TEST %d [%s]  100%%  loss = %.3f" % (elapsed_min, epoch + 1, "#" * 20, avg_loss))
    print("epoch %d complete, loss = %.03f ppl = %7.1f" % (epoch + 1, avg_loss, ppl))
    print(" ")

    return ppl

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
        for idx in tdqm(range(0,len(opt.train_labels),opt.batchsize), desc=f"Batches"):
            #define batch
            batch_examples = opt.train_tokenized[idx: idx + opt.batchsize]
            batch_labels = torch.tensor(opt.train_labels[idx: idx + opt.batchsize]).to(opt.device)
            #flatten for gpu pass
            allexamples = [torch.tensor(option[1:]) for example in batch_examples for option in example]
            choicestart = torch.tensor([option[0] for example in batch_examples for option in example], device=opt.device)
            #add padding so everything is the same legnth
            input_ids = torch.nn.utils.rnn.pad_sequence(allexamples, batch_first=True, padding_value=0).to(opt.device)
            attention_mask = (input_ids != 0).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x,logits = model(input_ids, attention_mask=attention_mask)
                #shift back by one so that model logits are at the same index as the tokens in the sequence
                logits = logits[:, :-1, :]
                labels = input_ids[:, 1:]
                log_probs = F.log_softmax(logits,dim=-1)
                #compute loss
                token_log_probs = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),labels.reshape(-1),reduction='none').view(labels.shape)
                positions = torch.arange(token_log_probs.size(1), device=opt.device).unsqueeze(0)
                mask = (positions >= (choicestart.unsqueeze(1) - 1)).float()
                token_log_probs = token_log_probs * mask
                choice_scores = token_log_probs.sum(dim=1)/mask.sum(dim=1)
                choice_scores = choice_scores.view(len(batch_examples), 4)
                loss = F.cross_entropy(choice_scores, batch_labels)
            opt.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt.optimizer)
            scaler.update()

        accuracy = test(model, opt, validation = True)
        if accuracy < prevaccuracy:
            #Early Stopping based on validation set
            break
        prevaccuracy = accuracy
        print("Accuracy is ", accuracy)
        #save model weights
        os.makedirs("GPTQAweights", exist_ok = True)
        intacc = int(accuracy * 100)
        filename = f"GPTQAweights/model_weights_zero_shot_{intacc}.pt" if opt.zero_shot else f"GPTQAweights/model_weights_{intacc}.pt"
        torch.save(model.state_dict(), filename)
        torch.cuda.empty_cache()

def test(model, opt, validation = False):
    examples = opt.valid_tokenized if validation else opt.test_tokenized
    alllabels = opt.valid_labels if validation else opt.test_labels
    model.eval()
    correct = 0
    with torch.no_grad():
        for  idx in tdqm(range(0,len(examples),opt.batchsize), desc=f"Evaluation on {"test" if not validation else "validation"} set"):
           #define batch
            batch_examples = examples[idx: idx + opt.batchsize]
            batch_labels = torch.tensor(alllabels[idx: idx + opt.batchsize]).to(opt.device)
            #flatten for single pass and move to gpu
            allexamples = [torch.tensor(option[1:]) for example in batch_examples for option in example]
            choicestart = torch.tensor([option[0] for example in batch_examples for option in example], device=opt.device)
            #flatten for gpu pass
            allexamples = [torch.tensor(option[1:]) for example in batch_examples for option in example]
            choicestart = torch.tensor([option[0] for example in batch_examples for option in example], device=opt.device)
            #add padding so everything is the same legnth
            input_ids = torch.nn.utils.rnn.pad_sequence(allexamples, batch_first=True, padding_value=0).to(opt.device)
            attention_mask = (input_ids != 0).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x,logits = model(input_ids, attention_mask=attention_mask)
                logits = logits[:, :-1, :]
                labels = input_ids[:, 1:]
                token_log_probs = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),labels.reshape(-1),reduction='none',).view(labels.shape)
                positions = torch.arange(token_log_probs.size(1), device=opt.device).unsqueeze(0)
                mask = (positions >= (choicestart.unsqueeze(1) - 1)).float()
                token_log_probs = token_log_probs * mask
                choice_scores = token_log_probs.sum(dim=1)/mask.sum(dim=1)
                choice_scores = choice_scores.view(len(batch_examples), 4)
                preds = choice_scores.argmax(dim=1)
                correct+=(preds==batch_labels).sum().item()
    return correct/len(alllabels)

#PART 3
class BeamNode(object):
    def __init__(self, logprob, tokensequence, depth):
        self.logprob = logprob
        self.tokensequence = tokensequence
        self.children = []
        self.depth = depth

class BeamTree(object):
    def __init__(self, model, stem, opt):
        self.activenodes = []
        self.root = BeamNode(logprob=0.0, depth=0, tokensequence=stem)
        self.model = model
        self.numbeams = opt.numbeams
        self.maxdepth = opt.beamlength
        self.activenodes = []
        self.temperature = opt.temperature
        self.device = opt.device
        self.repetition_penalty = opt.repetition_penalty
    def getnexttokens(self, tokenseq):
        with torch.no_grad():
            tokenseq = torch.tensor(tokenseq, device=self.device).unsqueeze(0)
            attention_mask = (tokenseq != 0).long()
            x,logits = self.model(tokenseq, attention_mask=attention_mask)
            #adj for temp
            log_probs = F.log_softmax(logits[:, -1, :] / self.temperature, dim=-1)
            for token in tokenseq:
                log_probs[0, token] /= self.repetition_penalty
            topk_log_probs, topk_ids = torch.topk(log_probs, k=self.numbeams)
            return topk_log_probs.squeeze(0), topk_ids.squeeze(0)

    def genbeams(self):
        #initial generation (working on root node)
        if not self.activenodes:
            probs, ids = self.getnexttokens(self.root.tokensequence)
            for prob, id in zip(probs, ids):
                self.activenodes.append(BeamNode(logprob=prob.item(), tokensequence=(self.root.tokensequence + [id.item()]), depth=1))
            return True
        #hit max depth -> stop gen
        elif self.activenodes[0].depth == self.maxdepth:
            return False
        else:
            newnodes = []
            for node in self.activenodes:
                probs, ids = self.getnexttokens(node.tokensequence)
                for prob, id in zip(probs, ids):
                    newnodes.append(BeamNode(logprob=(node.logprob + prob.item()), tokensequence=(node.tokensequence + [id.item()]), depth=node.depth+1))
            newnodes.sort(key=lambda node: node.logprob)
            self.activenodes = newnodes[-self.numbeams:]
            return True
    def buildtree(self):
        gen = True
        while gen:
            gen = self.genbeams()
        #ret top beam/sequence
        bestseq = torch.tensor(self.activenodes[-1].tokensequence,device=self.device).unsqueeze(0)
        attention_mask = (bestseq != 0).long()
        with torch.no_grad():
            embedding, _ = self.model(bestseq, attention_mask=attention_mask)
        return self.activenodes[-1].tokensequence, embedding.squeeze(0)

def Bertscore(choice, gen):
    choice = F.normalize(choice, p=2, dim=-1)
    gen = F.normalize(gen, p=2, dim=-1)
    similarity = torch.matmul(choice, gen.T)
    precision = similarity.max(dim=0).values.mean()
    recall = similarity.max(dim=1).values.mean()
    return 2 * precision * recall / (precision + recall)

def test_generative(model,opt,validation=False):
    correct = 0
    incorrect = 0

    exs = opt.valid_tokenized if validation else opt.test_tokenized
    alllabels = opt.valid_labels if validation else opt.test_labels

    examples = {"correct": [], "incorrect": []}
    for questionidx in tdqm(range(0,len(exs))):
        question = exs[questionidx]
        stemendidx = question[0][0]
        stem = question[0][1:stemendidx]
        choices = [torch.tensor(choice[choice[0]:]) for choice in question]
        tree = BeamTree(model=model,stem=stem, opt=opt)
        gen, embeddings = tree.buildtree()
        #get choice embeddings
        input_ids = torch.nn.utils.rnn.pad_sequence(choices, batch_first=True, padding_value=0).to(opt.device)
        attention_mask = (input_ids != 0).long()
        scores = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,):
            x, _ = model(input_ids, attention_mask=attention_mask)
            for choiceembed, choicemask in zip(x, attention_mask):
                scores.append(Bertscore(choiceembed[choicemask.bool()],embeddings[stemendidx:]))
        pred = torch.argmax(torch.stack(scores)).item()
        if pred==alllabels[questionidx]:
            if correct<5:
                answer = {"stem": opt.tokenizer.decode(stem),
                        "generation": opt.tokenizer.decode(gen[stemendidx:]),
                        "choices": [(opt.tokenizer.decode(choices[choiceidx]), scores[choiceidx].item()) for choiceidx in range(len(choices))],
                        "Correct answer choice": opt.tokenizer.decode(choices[alllabels[questionidx]].tolist())
                            }
                examples["correct"].append(answer)
            correct +=1
        elif incorrect < 5:
            incorrect+=1
            answer = {"stem": opt.tokenizer.decode(stem),
                        "generation": opt.tokenizer.decode(gen[stemendidx:]),
                        "choices": [(opt.tokenizer.decode(choices[choiceidx]), scores[choiceidx].item()) for choiceidx in range(len(choices))],
                        "Correct answer choice": opt.tokenizer.decode(choices[alllabels[questionidx]].tolist())
                            }
            examples["incorrect"].append(answer)
    return correct/(len(alllabels)), examples

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-loadname", type=str, default="")
    parser.add_argument("-valid_file", type=str, default="")
    parser.add_argument("-tokenizer_dir", type=str, default="")
    parser.add_argument("-d_model", type=int, default=1024)
    parser.add_argument("-d_ff", type=int, default=4096)
    parser.add_argument("-n_layers", type=int, default=16)
    parser.add_argument("-heads", type=int, default=16)
    parser.add_argument("-seqlen", type=int, default=1024)
    parser.add_argument("-eval_seqlen", type=int, default=None)
    parser.add_argument("-eval_batchsize", type=int, default=1)
    parser.add_argument("-dropout", type=float, default=0.0)
    parser.add_argument("-epsilon", type=float, default=1e-5)
    parser.add_argument("-no_cuda", action="store_true")
    #part 2 training
    parser.add_argument("-zero_shot", type=int, default=0)
    parser.add_argument("-batchsize", type=int, default=1)
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-lr", type=float, default=0.00001)
    #part 3 params
    parser.add_argument("-generativeanswering", type=int, default=0)
    parser.add_argument("-numbeams", type=int, default=1)
    parser.add_argument("-beamlength", type=int, default=1)
    parser.add_argument("-temperature", type=float, default=1.0)
    parser.add_argument("-repetition_penalty", type=float, default=1.0)

    opt = parser.parse_args()
    opt.zero_shot = False if opt.zero_shot == 0 else True
    generative = False if opt.generativeanswering == 0 else True

    opt.train = read_obqa('obqa/obqa.train.txt')
    opt.test = read_obqa('obqa/obqa.test.txt')
    opt.valid = read_obqa('obqa/obqa.valid.txt')

    if opt.eval_seqlen is None:
        opt.eval_seqlen = opt.seqlen

    device = torch.device("cuda:0" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    tokenizer = GPT2TokenizerFast.from_pretrained(opt.tokenizer_dir)
    opt.tokenizer = tokenizer
    opt.device = device

    #precompute embeddings
    Tokenize(opt)

    state_dict = load_model_best_state_dict(opt.loadname)
    vocab_size = int(state_dict["wte.weight"].shape[0])

    model = TransformerGPT(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.seqlen, opt.d_ff,
                           opt.dropout, opt.epsilon)
    model.load_state_dict(state_dict, strict=True)

    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    model.train()
    #part 2 (decoder only training)
    if not generative:
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
        train(model,opt)
        accuracy = test(model, opt)
        print("Test set accuracy for decoder only is: ", accuracy)

    #part 3 (load the finetune from part 2)
    else:
        accuracy, examples = test_generative(model, opt, validation=True)
        print("Validtion set accuract for generative -> berstscore classification is: ", accuracy)
        accuracy, examples = test_generative(model, opt, validation=False)
        print("Test set accuract for generative -> berstscore classification is: ", accuracy)
        print("generation examples are", examples)

    # indices = my_tokenizer(opt.valid_file,tokenizer,1000000)
    # ppl = test_model(model=model, indices=indices, opt=opt, epoch=0, device=device)

if __name__ == "__main__":
    main()