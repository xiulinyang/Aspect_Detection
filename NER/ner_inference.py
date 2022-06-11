import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda
from seqeval.metrics import classification_report
from ner import *
# from sklearn.metrics import classification_report
device = 'cuda' if cuda.is_available() else 'cpu'

def inference(test_data, model, tokenizer, ids_to_labels,config):
    truth_tokens, predictions = [], []

    for idx, row in test_data.iterrows():
        # print(row['tokens'])
        inputs = tokenizer(row['tokens'],
                            is_split_into_words=True, 
                            return_offsets_mapping=True, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=config['MAX_LEN'],
                            return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        # forward pass
        outputs = model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
            print(token_pred, mapping)
            #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue

        truth_tokens.append(row['tokens'])
        predictions.append(prediction)

    return truth_tokens, predictions



def get_index(predictions):
    total_idx = []
    for labels in predictions:
        # print(len(labels))
        phrase_index = []
        count = 0
        i_flag=False
        for j in range(len(labels)):
            # print(j)
            if labels[j] == 'B-ASP' or (labels[j]=='I-ASP' and i_flag==False):
                i_flag = False
                idx = []
                dc = {}
                count += 1
                idx.append(j)
                for k in range(j+1,len(labels)):
                    if labels[k] == 'I-ASP':
                        i_flag = True
                        idx.append(k)
                    else:
                        break
                dc[count] = idx
                phrase_index.append(dc)
        total_idx.append(phrase_index)
    return total_idx


def get_aspect(truth_tokens, total_idx):
    pred_aspect = []
    for i in range(len(total_idx)):
        aspect = []
        if len(total_idx[i]) > 0:
            for idx in total_idx[i]:  
                if len(idx.keys()) > 0:
                    token_idx = list(idx.values())[-1]
                    print(token_idx)
                    if len(token_idx) == 1:
                        aspect.append(truth_tokens[i][token_idx[0]])
                    else:
                        aspect.append(' '.join(truth_tokens[i][token_idx[0]: token_idx[-1]+1]))

        pred_aspect.append(aspect)
    
    return pred_aspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type = str, help='Path to train data.', default='tagged_ner_test.json')
    parser.add_argument('--model_dir', type = str, help='Path to Pretrained model', default = 'Models_1.0')
    args = parser.parse_args()

    model_dir = args.model_dir
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model.cuda()

    config =  {
    'MAX_LEN' : 128,
    'TEST_BATCH_SIZE': 32,
    }



    test_data = pd.read_json('./tagged_ner_test.json')

    labels_to_ids, ids_to_labels = label_ids(test_data)

    test_set = dataset(test_data, tokenizer, config['MAX_LEN'], labels_to_ids)


    test_parms = {'batch_size': config['TRAIN_BATCH_SIZE'],
                'shuffle': True,
                'num_workers': 0
                }


    test_loader = DataLoader(test_set, **test_parms)

    eval_loss, eval_accuracy = valid(model, test_loader)

    truth_tokens, predictions = inference(test_data, model, tokenizer, ids_to_labels, config)

    token_index = get_index(predictions)

    pred_aspect = get_aspect(truth_tokens, token_index)

    with open('pred_aspect.txt', 'w') as f:
        for asp in pred_aspect:
            f.write(asp)
            f.write("\n")
        
        f.close()



