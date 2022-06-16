import argparse
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast, BertForTokenClassification
from torch import cuda
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt


import seaborn as sns
# from sklearn.metrics import classification_report
torch.cuda.empty_cache()

device = 'cuda' if cuda.is_available() else 'cpu'

def label_ids(df):
  total_tags = []
  for index, row in df.iterrows():
    total_tags.extend((row['tags']))

  total_tags = np.unique(np.array(total_tags))
  labels_to_ids , ids_to_labels = {}, {}
  for idx, v in enumerate(total_tags):
    labels_to_ids[v] = idx
    ids_to_labels[idx] = v

  return labels_to_ids , ids_to_labels



class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.tokens[index] 
        word_labels = self.data.tags[index]
        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.labels_to_ids[label] for label in word_labels]

        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1


        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len



def valid(model, valid_loader, ids_to_labels):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(valid_loader)):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            loss, eval_logits = output[0], output[1]
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            # if idx % 100==0:
            #     loss_step = eval_loss/nb_eval_steps
            #     print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    # print(len(labels))
    # print(len(predictions))


    print(classification_report([labels], [predictions]))

    return eval_loss, eval_accuracy

# Defining the training function on the 80% of the dataset for tuning the bert model
def train(model, train_loader, valid_loader, optimizer, tokenizer, output_dir, config, ids_to_labels):
    valid_acc = 0
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    training_stats = []
    # put model in training mode
    model.train()
    patience = 0
    for i in range(config['EPOCHS']):
      if patience <= config['PATIENCE']:
        for idx, batch in tqdm(enumerate(train_loader)):
          ids = batch['input_ids'].to(device, dtype = torch.long)
          mask = batch['attention_mask'].to(device, dtype = torch.long)
          labels = batch['labels'].to(device, dtype = torch.long)

          outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
          loss, tr_logits = outputs[0], outputs[1]
          tr_loss += loss.item()

          nb_tr_steps += 1
          nb_tr_examples += labels.size(0)
          
            
          # compute training accuracy
          flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
          active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
          flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
          
          # only compute accuracy at active labels
          active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
          #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
          
          labels = torch.masked_select(flattened_targets, active_accuracy)
          predictions = torch.masked_select(flattened_predictions, active_accuracy)
          
          tr_labels.extend(labels)
          tr_preds.extend(predictions)

          tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
          tr_accuracy += tmp_tr_accuracy
      
          # gradient clipping
          torch.nn.utils.clip_grad_norm_(
              parameters=model.parameters(), max_norm=config['MAX_GRAD_NORM']
          )
          
          # backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

        eval_loss, eval_accuracy = valid(model, valid_loader, ids_to_labels)
        patience += 1

        if valid_acc < eval_accuracy:
          patience = 0
          valid_acc = eval_accuracy
          print('Saving Model')
          model.save_pretrained(output_dir)
          tokenizer.save_pretrained(output_dir)


            # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': i + 1,
                'Training Loss': epoch_loss,
                'Valid. Loss': eval_loss,
                'Valid. Accur.': eval_accuracy
            }
        )
      else:
        print("Early Stopping")
        break

    return model, training_stats





def load_data(train_path, val_path):
  train_data = pd.read_json(train_path)
  val_data = pd.read_json(val_path)
  print(len(train_data), len(val_data))
  return train_data, val_data





def plot_history(df_stats):
  # Use plot styling from seaborn.
  sns.set(style='darkgrid')

  # Increase the plot size and font size.
  sns.set(font_scale=1.5)
  plt.rcParams["figure.figsize"] = (12,6)

  # Plot the learning curve.
  plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
  plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

  # Label the plot.
  plt.title("Training & Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.xticks([1, 2, 3, 4])

  plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', type = str, help='Path to train data.', default='Data/tagged_ner_train.json')
  parser.add_argument('--val_path', type = str, help='Path to validation data.', default='Data/tagged_ner_dev.json')
  parser.add_argument('--output_dir',type = str , help='Path to save model', default = 'Models')
  parser.add_argument('--model', type = str, help='Choose a Pretrained model', default = 'base')
  parser.add_argument('--epoch', type = int, help='Number of epoch', default = 5)
  args = parser.parse_args()
  train_path = args.train_path
  val_path = args.val_path
  output_dir = args.output_dir
  if args.model=='base':
    model = 'bert-base-uncased'
  else:
    model = 'bert-large-uncased'

  

  config =  {
    'MAX_LEN' : 128,
    'TRAIN_BATCH_SIZE': 16,
    'VALID_BATCH_SIZE' : 32,
    'EPOCHS' : 5,
    'LEARNING_RATE' : 3e-05,
    'MAX_GRAD_NORM' : 10,
    'PATIENCE': 3
 
  }
  tokenizer = BertTokenizerFast.from_pretrained(model)
  
  train_data, val_data = load_data(train_path, val_path)

  labels_to_ids , ids_to_labels = label_ids(train_data)

  training_set = dataset(train_data, tokenizer, config['MAX_LEN'], labels_to_ids)
  valid_set = dataset(val_data, tokenizer, config['MAX_LEN'], labels_to_ids)

  train_params = {'batch_size': config['TRAIN_BATCH_SIZE'],
                'shuffle': True,
                'num_workers': 0
                }

  test_params = {'batch_size': config['VALID_BATCH_SIZE'],
                  'shuffle': True,
                  'num_workers': 0
                  }

  train_loader = DataLoader(training_set, **train_params)
  valid_loader = DataLoader(valid_set, **test_params)

  model = BertForTokenClassification.from_pretrained(model, num_labels=len(labels_to_ids))
  model.to(device)

  optimizer = torch.optim.Adam(params=model.parameters(), lr=config['LEARNING_RATE'])

  model, training_stats = train(model, train_loader, valid_loader, optimizer, tokenizer, output_dir, config, ids_to_labels)

  # Display floats with two decimal places.
  pd.set_option('precision', 2)

  # Create a DataFrame from our training statistics.
  df_stats = pd.DataFrame(data=training_stats)

  # Use the 'epoch' as the row index.
  df_stats = df_stats.set_index('epoch')


  plot_history(df_stats)




if __name__ == '__main__':
  main()