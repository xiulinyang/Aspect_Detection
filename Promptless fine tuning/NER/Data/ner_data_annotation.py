import pandas as pd
import re
import json

train_dir = 'argument_aspect_detection/in_topic/train.jsonl'
dev_dir = 'argument_aspect_detection/in_topic/dev.jsonl'
test_dir = 'argument_aspect_detection/in_topic/test.jsonl'
data_train = pd.read_json(train_dir, lines=True)  
data_dev = pd.read_json(dev_dir, lines=True)
data_test = pd.read_json(test_dir, lines=True)


ind_train = data_train.aspect_pos.apply(lambda x: 'no_Aspect' not in x)
ind_dev = data_dev.aspect_pos.apply(lambda x: 'no_Aspect'  not in x)
ind_test = data_test.aspect_pos.apply(lambda x: 'no_Aspect' not in x)


data_train = data_train[ind_train]
data_dev = data_dev[ind_dev]
data_test =data_test[ind_test]

print('dataset \t\t\t number of no_aspect\t\t\tnumber of all sentences')
print('-'*100)
print(f'train \t\t\t\t {len(data_train)} \t\t\t\t {len(ind_train)}')
print(f'dev \t\t\t\t {len(data_dev)} \t\t\t\t {len(ind_dev)}')
print(f'test \t\t\t\t {len(data_dev)} \t\t\t\t {len(ind_dev)}')



def get_data(keyword, df=data_train):
    data = df[[keyword]].values.tolist()
    data = [sample for samples in data for sample in samples]
    return data


X_train = get_data('sentence')
X_train_asp_pos = get_data('aspect_pos')
X_train_asp_str = get_data('aspect_pos_string')

X_dev = get_data('sentence', data_dev)
X_dev_asp_pos = get_data('aspect_pos', data_dev)
X_dev_asp_str = get_data('aspect_pos_string', data_dev)

X_test = get_data('sentence', data_test)
X_test_asp_pos = get_data('aspect_pos', data_test)
X_test_asp_str = get_data('aspect_pos_string', data_test)


def get_asp_pos(split_asp_pos):
    pos_list =[]
    for sent in split_asp_pos:
        sent_list =[]
        sent_list.append(0)
        for word in sent:
            nums = re.findall(r'\d+', word)
            for i, num in enumerate(nums):
                if i%2==0:
                    sent_list.append(int(num))
                else:
                    sent_list.append(int(nums[i-1])+int(num))
        pos_list.append(sent_list) 
    return pos_list

train_asp_pos = get_asp_pos(X_train_asp_pos)
dev_asp_pos = get_asp_pos(X_dev_asp_pos)
test_asp_pos = get_asp_pos(X_test_asp_pos)


    
    
def get_tag(X_split,split_asp_pos):
    data = get_data('sentence', X_split)
    tag_data=[]
    for i, sent in enumerate(data):
        sent_dic ={}
        for j in range(len(split_asp_pos[i])-1):
            if j%2!=0:
                sent_dic[sent[split_asp_pos[i][j]:split_asp_pos[i][j+1]]] = 'I-ASP' 
            else:
                sent_dic[sent[split_asp_pos[i][j]:split_asp_pos[i][j+1]]] ='O'
        tag_data.append(sent_dic)
    
    single_tagged =[]
    for sample in tag_data:
        single=[]
        for key, value in sample.items():
            keys = key.rstrip().split(' ')
            for i, k in enumerate(keys):
                if i==0 and value =='I-ASP':
                    single.append((k, 'B-ASP'))
                else:
                    single.append((k, value))
        single_tagged.append(single)
    return single_tagged


train_tag = get_tag(data_train, train_asp_pos)
dev_tag = get_tag(data_dev, dev_asp_pos)
test_tag = get_tag(data_test, test_asp_pos)

print(X_train[0], train_tag[0], X_train_asp_str[0])

def generate_file(file_name, ner_label, sent_split, aspects):
    with open (file_name, 'w') as f:
        data = list()
        for i, sent in enumerate(ner_label):
            dc = dict()
            dc['sentence'] = sent_split[i]
            dc['tokens'] = list(touple[0] for touple in ner_label[i])
            dc['tags'] = list(touple[1] for touple in ner_label[i])
            dc['aspects'] = aspects[i]
            data.append(dc)    
        json_object = json.dumps(data, indent = 4)
        f.write(json_object)

generate_file('tagged_ner_train.json', train_tag, X_train, X_train_asp_str)
generate_file('tagged_ner_dev.json', dev_tag, X_dev, X_dev_asp_str)
generate_file('tagged_ner_test.json', test_tag, X_test, X_test_asp_str)
