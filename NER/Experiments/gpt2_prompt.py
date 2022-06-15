# -*- coding: utf-8 -*-
"""GPT2_prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MC5buRx3fTkiebyj-Zw_G98wP8-yqqRs
"""

!pip install transformers

!nvidia-smi

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import re

"""Two questions: (1) should we stem the label before training? (2) how should we get the ids of the labels"""

def template(sentence, aspects=None):
    if aspects is None:
        tem = f" Sentence: {sentence} Aspect: "
    else:
        tem = f" Sentence: {sentence} Aspect: {aspects}."
    return tem

def generateTemplate(data):
    train_idxs = np.random.randint(0, len(data) - 1, 3)
    for i, sent in enumerate(data):
      text = ""
      for ti in train_idxs:
          sentence = data.iloc[ti]["sentence"]
          sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
          aspects = ", ".join(data.iloc[ti]["aspect_pos_string"])
#         aspects = train_df.iloc[ti]["topic"]
          text += template(sentence, aspects)
    
          text += template(data.iloc[i]["sentence"])
          aspects = ", ".join(data.iloc[i]["aspect_pos_string"])
#     aspects = test_df.iloc[test_idxs]["topic"]
    return text, aspects

train = pd.read_json('./in_topic/train.jsonl', lines =True)
dev = pd.read_json('./in_topic/dev.jsonl', lines =True)
test= pd.read_json('./in_topic/test.jsonl', lines =True)

class ArgumentData:
  def __init__(self, tokenizer, data, max_length=512):
    self.input_ids =[]
    self.attention_mask =[]
    self.aspect =[]
    for i, sent in enumerate(data['sentence']):
      train_idxs = np.random.randint(0, len(data)-1, 3)
      text = ""
      for ti in train_idxs:
          sentence = data.iloc[ti]["sentence"]
          sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
          aspects = ", ".join(data.iloc[ti]["aspect_pos_string"])
          text += template(sentence, aspects)
    
      text += template(data.iloc[i]["sentence"], ', '.join(data.iloc[i]['aspect_pos_string']))
      encoding_dic = tokenizer(text, truncation=True, max_length=max_length, padding='max_length')
      self.input_ids.append(torch.tensor(encoding_dic['input_ids']))
      self.attention_mask.append(torch.tensor(encoding_dic['attention_mask']))
      self.aspect.append(aspects)
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.attention_mask[idx], self.aspect[idx]

train_dir = '/content/gdrive/MyDrive/in_topic/train.jsonl'
dev_dir = '/content/gdrive/MyDrive/in_topic/dev.jsonl'
test_dir ='/content/gdrive/MyDrive/in_topic/test.jsonl'
def data_load(dir, tokenizer):
  data = pd.read_json(dir, lines=True)
  processed_data = ArgumentData(tokenizer,data,max_length=512)
  return processed_data

model_name ='gpt2'
torch.manual_seed(123)
tokenizer =GPT2Tokenizer.from_pretrained(model_name, bos_token='<startoftext>', eos_token ='<endoftext>', pad_token='<pad>')
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

train_dataset = data_load(train_dir, tokenizer=tokenizer)
test_dataset = data_load(test_dir, tokenizer)
dev_dataset = data_load(dev_dir, tokenizer)

train_dataset = train_dataset

from transformers import Trainer, TrainingArguments
torch.cuda.empty_cache()

training_args= TrainingArguments(output_dir='/content/gdrive/MyDrive/in_topic/results', num_train_epochs=2, logging_steps=100,
                                  save_strategy='epoch', 
                                 per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                 warmup_steps=100, weight_decay=0.01, logging_dir='/content/gdrive/MyDrive/in_topic/logs')
Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, 
        data_collator=lambda data:{'input_ids': torch.stack([f[0] for f in data]),
                                   'attention_mask': torch.stack([f[1] for f in data]),
                                   'labels': torch.stack([f[0] for f in data])}).train()

from nltk.translate.bleu_score import sentence_bleu

model.eval()
test_dataset = pd.read_json(test_dir, lines=True)

test_dt =[]
test_aspect =[]
for i, sent in enumerate(test_dataset['sentence']):
      train_idxs = np.random.randint(0, len(test_dataset)-1, 3)
      text = ""
      for ti in train_idxs:
          sentence = test_dataset.iloc[ti]["sentence"]
          sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
          aspects = ", ".join(test_dataset.iloc[ti]["aspect_pos_string"])
          text += template(sentence, aspects)
      text += template(test_dataset.iloc[i]["sentence"])
      test_dt.append(text)
      tst_aspect =', '.join(test_dataset.iloc[i]['aspect_pos_string'])
      test_aspect.append(tst_aspect)

scores=[]
for text, label in tqdm(zip(test_dt, test_aspect)):
  prompt = text
  generated = tokenizer(f'<startoftext>{prompt}', return_tensors='pt').input_ids.cuda()
  sample_outputs = model.generate(generated, do_sammple=False, top_k=5, max_length=512, top_p = 0.5, temperature=0.9, num_return_sequences=1)
  predicted_text = tokenizer.decode(sample_outputs[0], skip_special_token=True)

  newtext = predicted_text.replace(test_dt, "")
  newaspect = newtext.split(".")[0]
  score = sentence_bleu(aspect.split(), newaspect.split())
  scores.append(score)