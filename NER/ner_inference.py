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
from Evaluation import *
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
            # print(token_pred, mapping)
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
                    # print(token_idx)
                    if len(token_idx) == 1:
                        aspect.append(truth_tokens[i][token_idx[0]])
                    else:
                        aspect.append(' '.join(truth_tokens[i][token_idx[0]: token_idx[-1]+1]))

        pred_aspect.append(aspect)
    
    return pred_aspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type = str, help='Path to train data.', default='Data/tagged_ner_test.json')
    parser.add_argument('--model_dir', type = str, help='Path to Pretrained model', default = './Models/Models_2.2')
    parser.add_argument('--score', type = str, help='Path to result.', default='bleu')
    args = parser.parse_args()

    model_dir = args.model_dir
    test_path = args.test_path
    score = args.score
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model.cuda()

    config =  {
    'MAX_LEN' : 128,
    'TEST_BATCH_SIZE': 16,
    }



    test_data = pd.read_json(test_path)

    labels_to_ids, ids_to_labels = label_ids(test_data)

    test_set = dataset(test_data, tokenizer, config['MAX_LEN'], labels_to_ids)


    test_parms = {'batch_size': config['TEST_BATCH_SIZE'],
                'shuffle': True,
                'num_workers': 0
                }


    test_loader = DataLoader(test_set, **test_parms)

    eval_loss, eval_accuracy = valid(model, test_loader, ids_to_labels)

    truth_tokens, predictions = inference(test_data, model, tokenizer, ids_to_labels, config)

    token_index = get_index(predictions)

    pred_aspect = get_aspect(truth_tokens, token_index)

    pred_aspect = [','.join(asp_list) for asp_list in pred_aspect]


    true_aspect = test_data['aspects']

    true_aspect = [','.join(asp) for asp in true_aspect]

    # print(pred_aspect[:20],true_aspect[:10])

    # d = {"predictions":pred_aspect,
    #     "references": true_aspect}

    

    # with open(output_file, 'w') as f:
    #     json.dump(d, f)


    if score =='bleu':
       score = get_bleu_score(pred_aspect, true_aspect) 
    elif score == 'bert':
        score = get_bert_score(pred_aspect, true_aspect)
    else:
        print("Invalid Score Argument")
        
if __name__ == '__main__':
    main()
    



