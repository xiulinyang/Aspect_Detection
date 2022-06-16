import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import re
from sacrebleu.metrics import BLEU
import bert_score
from bert_score import score
import argparse
import numpy as np
from tqdm import tqdm, trange


def template(sentence, aspects=None):
    if aspects is None:
        tem = f" Sentence: {sentence} Aspect: "
    else:
        tem = f" Sentence: {sentence} Aspect: {aspects}."
    return tem
    
    
def generateTemplate(data_dir, num):
    data =pd.read_json(data_dir, lines=True)
    train_idxs = np.random.randint(0, len(data) - 1, num)
    for i, sent in enumerate(data):
      text = ""
      for ti in train_idxs:
          sentence = data.iloc[ti]["sentence"]
          sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
          aspects = ", ".join(data.iloc[ti]["aspect_pos_string"])
#         aspects = train_df.iloc[ti]["topic"]
          text += template(sentence, aspects)
    return text
    
    

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

def get_aspect(pred_dir):
  aspect_list =[]
  with open(pred_dir, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if '<startoftext>' in line:
        line = re.sub(r'<startoftext>', '', line)
        line = re.sub('�', '', line)
        line = line.split('Aspect: ')[1].split('.')[0]
        aspect_list.append(line.strip())
    return aspect_list


    
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type = str, help='Path to test.', default='Data/test.jsonl')
    parser.add_argument('--model_dir', type = str, help='Path to Pretrained model')
    parser.add_argument('--score', type = str, help='evaluation method including bert and bleu.', default='bleu')
    parser.add_argument('--number_prompt', type=int, help ='the number of prompt preceding the test sentence', default=3)
    parser.add_argument('--model_name', type=str, help='The name of the LM including gpt2 and gptneo', default='gpt2')
    args = parser.parse_args()

    model_dir = args.model_dir
    train_path = './in_topic/train.jsonl'
    test_path = args.test_path
    score = args.score
    num = args.number_prompt
    model_name = args.model_name
    
    if model_name =='gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    if model_name =='gptneo':
        model = GPTNeoForCausalLM.from_pretrained(model_dir)
        
    else:
        print('This language model is not available yet.')
    tokenizer =GPT2Tokenizer.from_pretrained('gpt2', bos_token='<startoftext>', eos_token ='<endoftext>', pad_token='<pad>')
    model.to(device='cuda')
    
    test_dataset = pd.read_json(test_path, lines=True)


    
    test_dt =[]
    test_aspect =[]
    for i, sent in enumerate(test_dataset['sentence'].to_list()):
        text = ""
        tem = generateTemplate(train_path,num)
        text +=tem
        text += template(sent)
        test_dt.append(text.strip())
        tst_aspect =', '.join(test_dataset.iloc[i]['aspect_pos_string'])
        test_aspect.append(tst_aspect)
        print(i, text)  
    
    
    with open ('fixed_prompt_tuning_prompt2.txt', 'w') as f:
        for text, label in tqdm(zip(test_dt, test_aspect)):
            prompt = text
            generated = tokenizer(f'<startoftext>{prompt}', return_tensors='pt').input_ids.cuda()
            sample_outputs = model.generate(generated, do_sample=False, top_k=5, max_length=512, top_p = 0.5, temperature=0.7, num_return_sequences=1)
            predicted_text = tokenizer.decode(sample_outputs[0], skip_special_token=True)
            f.write(f'prompt: {prompt} \n')
            f.write(f'correct aspects: {label} \n')
            f.write(f'predicted aspects: {predicted_text} \n\n\n')
            
    predicted_aspect = get_aspect('fixed_prompt_tuning_prompt2.txt')
    
    if score =='bleu':
        score = get_bleu_score(predicted_aspect, test_aspect)
    if score =='bert':
        score = get_bert_score(predicted_aspect, test_aspect)
    else:
        print('invalid evaluation method')
    
    print(score)
    
if __name__ == '__main__':
    main()
    


    
    
    
    
    
    
    
    
    
    
    
    