# -*- coding: utf-8 -*-
import argparse
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
from transformers import Trainer, TrainingArguments
torch.manual_seed(123)

def template(sentence, aspects=None):
    if aspects is None:
        tem = f" Sentence: {sentence} Aspect: "
    else:
        tem = f" Sentence: {sentence} Aspect: {aspects}."
    return tem

def generateTemplate(data):
    for i, sent in enumerate(data):
      text = ""
      text += template(data.iloc[i]["sentence"])
      aspects = ", ".join(data.iloc[i]["aspect_pos_string"])
#     aspects = test_df.iloc[test_idxs]["topic"]
    return text, aspects

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
    
def data_load(dir, tokenizer):
  data = pd.read_json(dir, lines=True)
  processed_data = ArgumentData(tokenizer,data,max_length=512)
  return processed_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type = str, help='Path to train data.', default='Data/train.jsonl')
    parser.add_argument('--val_path', type = str, help='Path to validation data.', default='Data/dev.jsonl')
    parser.add_argument('--output_dir',type = str , help='Path to save model')
    parser.add_argument('--model', type = str, help='Choose a Pretrained model', choices ={"gpt2", 'gptneo'})
    parser.add_argument('--epoch', type = int, help='Number of epoch', default = 5)
  
    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    output_dir = args.output_dir
    
    if args.model=='gpt2':
        model_name = 'gpt2'
    if args.model =='gptneo':
        model_name = 'EleutherAI/gpt-neo-1.3B'
    
      

    train_dir = './in_topic/train.jsonl'
    dev_dir = './in_topic/dev.jsonl'
    test_dir ='./in_topic/test.jsonl'



    tokenizer =GPT2Tokenizer.from_pretrained('gpt2', bos_token='<startoftext>', eos_token ='<endoftext>', pad_token='<pad>')
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    train_dataset = data_load(train_dir, tokenizer)
    test_dataset = data_load(test_dir, tokenizer)
    dev_dataset = data_load(dev_dir, tokenizer)



    torch.cuda.empty_cache()

    training_args= TrainingArguments(output_dir='./in_topic/results', num_train_epochs=5, logging_strategy ='epoch', 
                                        save_strategy='epoch', evaluation_strategy = 'epoch', per_device_train_batch_size=2, 
                                        per_device_eval_batch_size=2, warmup_steps=100, weight_decay=0.01, logging_dir='./in_topic/logs')
    Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, 
            data_collator=lambda data:{'input_ids': torch.stack([f[0] for f in data]),
                                  'attention_mask': torch.stack([f[1] for f in data]),
                                  'labels': torch.stack([f[0] for f in data])}).train()

if __name__ == '__main__':
    main()
    