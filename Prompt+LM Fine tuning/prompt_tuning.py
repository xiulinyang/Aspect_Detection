import os
import gc
import copy
import random
import datetime
import time
import string
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser

from sacrebleu.metrics import BLEU
from bert_score import score

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

from nltk.translate.bleu_score import sentence_bleu
import re

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class PrompDataset(Dataset):

    def __init__(self, X, tokenizer):
        self.X = X
        self.tokenizer = tokenizer

    def getTemplate(self, sentence, aspect):
        prompt = f"{SPECIAL_TOKENS['bos_token']}{sentence}{aspect}{SPECIAL_TOKENS['eos_token']}"
        prompt = f"{sentence}: {aspect}."
        return prompt

    def processText(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = text.replace("  ", " ")
        return text

    def __getitem__(self, idx):
        sentence = self.X[idx][-1]
        sentence = self.processText(sentence)

        aspect = sentence + ", ".join(self.X[idx][2])
        template = self.getTemplate(sentence, aspect)

        sentence_dict = self.tokenizer(template, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        # aspect_dict = self.tokenizer(aspect, max_length=MAXLEN, padding="max_length", return_tensors="pt")

        # label = torch.tensor(aspect_dict["input_ids"])
        input_ids = torch.tensor(sentence_dict["input_ids"])
        attention_mask = torch.tensor(sentence_dict["attention_mask"])

        return {"label": input_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask}

    def __len__(self):
        return self.X.shape[0]

class PROMPTEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(wte.weight.size(1), n_tokens).uniform_(-random_range, random_range)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


def getInputs(prompt, word=None):
    if word:
        prompt += word

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs['input_ids'] = torch.cat([torch.full((1, NTOKENS), 50256), inputs['input_ids']], 1).to(DEVICE)
    inputs['attention_mask'] = torch.cat([torch.full((1, NTOKENS), 1), inputs['attention_mask']], 1).to(DEVICE)

    return inputs


def getDecoded(outputs):
    tokout = outputs.logits[0][-1].argmax()
    return tokenizer.decode(tokout, skip_special_tokens=True)


def template(sentence, aspect=""):
    return f"Sentence:{sentence}. Aspect:{aspect}"


def preprocessText(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = text.replace("  ", " ")
    return text


def promptLoader(test_dataset):
    test_dt = []
    test_aspect = []
    test_sentence = []
    for i, sent in enumerate(test_dataset['sentence']):
        train_idxs = np.random.randint(0, len(test_dataset) - 1, 3)
        text = ""
        for ti in train_idxs:
            sentence = test_dataset.iloc[ti]["sentence"]
            sentence = preprocessText(sentence)

            aspectlist = [preprocessText(txt) for txt in test_dataset.iloc[ti]["aspect_pos_string"]]
            aspects = ", ".join(aspectlist) + "."

            text += template(sentence, aspects)

        sentence = test_dataset.iloc[i]["sentence"]
        sentence = preprocessText(sentence)
        test_sentence.append(sentence)
        text += template(sentence)
        test_dt.append(text)
        tst_aspect = ', '.join(test_dataset.iloc[i]['aspect_pos_string'])
        test_aspect.append(tst_aspect)

    return test_dt, test_aspect, test_sentence


def cleanAspect(final_sentect, prompt):
    aspect = final_sentence[len(prompt):].split(".")[0]
    for key in SPECIAL_KEYS:
        aspect = aspect.replace(key, "")

    return aspect

def get_bleu_score(pred, test):
    '''
    This function calculate the blue score for the prediction of the model
    Argument:
    pred: the prediction of the model which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    test: the ground truth of the target labels which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    Output: Bleu Score Bleu Score for unigram/Bleu Score for bigram/Bleu Score for trigram
    '''
    test = [test]
    bleu = BLEU(max_ngram_order=3)
    score = bleu.corpus_score(pred,test)
    return score

def get_bert_score(pred, test):
    '''
    This function calculate the BERT score for the prediction of the model
    Argument:
    pred: the prediction of the model which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    test: the ground truth of the target labels which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    Output: The list of precision scores, recall scores, and F1 scores for each of the predictions;
    the output can be used for more detailed error analysis.
    '''
    P, R, F1 = score(pred, test, lang='en',model_type="microsoft/deberta-xlarge-mnli",  verbose=True)
    print(f'The average Bert Precision score of the prediction is {P.mean()}')
    print(f'The average Bert Recall score of the prediction is {R.mean()}')
    print(f'The average Bert F1 score of the prediction is {F1.mean()}')
    return P, R, F1


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Options: 'neo' for GPT-neo & 'gpt2' for GPT-2")
    args = parser.parse_args()

    SEED = 42
    seed_everything(SEED)

    MAXLEN = 512
    EPOCHS = 10
    SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                      "eos_token": "<|EOS|>",
                      "unk_token": "<|UNK|>",
                      "pad_token": "<|PAD|>",
                      "sep_token": "<|SEP|>"}
    SPECIAL_KEYS = list(SPECIAL_TOKENS.values())

    MAXLEN = 200
    MODEL_SAVE_PATH = "models"

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Creating model save path....")
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    print("Initializing tokenizer...")

    if args.model_name == "neo":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    elif args.model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    UNFREEZE_LAST_N = 6
    BATCH_SIZE = 4
    BATCH_UPDATE = 2
    NTOKENS = 10

    train_path = "data/train.jsonl"
    val_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"

    df = pd.read_json(train_path, lines=True)
    train_X = df.values

    df = pd.read_json(val_path, lines=True)
    val_X = df.values

    df = pd.read_json(test_path, lines=True)
    test_X = df.values


    train_dataset = PrompDataset(train_X, tokenizer)
    val_dataset = PrompDataset(val_X, tokenizer)
    test_dataset = PrompDataset(test_X, tokenizer)
    print("Prompt dataset loaded...")

    if args.model_name == "neo":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    elif args.model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    print(f"{args.model_name.upper()} Model initialized...")

    model.resize_token_embeddings(len(tokenizer))

    for p in model.parameters():
        p.requires_grad = False

    prompt_emb = PROMPTEmbedding(model.get_input_embeddings(),
                                 n_tokens=NTOKENS,
                                 initialize_from_vocab=True)

    model.set_input_embeddings(prompt_emb)

    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train_loss_curve = []
    val_loss_curve = []

    best_loss = 50

    print("Begin training.....")
    for epoch in range(EPOCHS):
        train_total_loss = 0
        val_total_loss = 0

        loss_idx = 0
        start_time = time.time()
        for data in tqdm(train_loader):
            input_ids = data["input_ids"].to(DEVICE)
            attention_mask = data["attention_mask"].to(DEVICE)
            # aspect_data = data["aspect_dict"]["input_ids"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            loss_idx += 1
            train_loss_avg = train_total_loss / loss_idx

        loss_idx = 0
        for data in tqdm(val_loader):
            input_ids = data["input_ids"].to(DEVICE)
            attention_mask = data["attention_mask"].to(DEVICE)
            # aspect_data = data["aspect_dict"]["input_ids"].to(DEVICE)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss

            val_total_loss += loss.item()
            loss_idx += 1
            val_loss_avg = val_total_loss / loss_idx

        if val_loss_avg < best_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f"{MODEL_SAVE_PATH}/bestmodel.pth")
            best_loss = val_loss_avg
            print(f"Saving best model at Epoch {epoch}....")

        print(
            f"EPOCH: {epoch}/{EPOCHS} | TRAIN LOSS: {train_loss_avg} | VAL LOSS: {val_loss_avg} TIME: {round(time.time() - start_time, 2)} seconds")
        train_loss_curve.append(train_loss_avg)
        val_loss_curve.append(val_loss_avg)

    test_df = pd.read_json("data/test.jsonl", lines=True)
    test_dt, test_apsect, test_sentence = promptLoader(test_df)

    scores = []
    n_gen = 15

    pred_aspects = []

    print("Begin Testing....")
    for prompt, label in tqdm(zip(test_dt, test_apsect)):
        final_sentence = prompt
        word = None
        for i in range(n_gen):
            inputs = getInputs(prompt, word)
            outputs = model(**inputs)
            word = getDecoded(outputs)
            final_sentence += word

        aspect = cleanAspect(final_sentence, prompt)
        score = sentence_bleu(label.split(), aspect)
        scores.append(score)
        pred_aspects.append(aspect)

    bleu_score = get_bleu_score(pred_aspects, test_apsect)
    bert_scores = get_bert_score(pred_aspects, test_apsect)
    print(bert_scores)