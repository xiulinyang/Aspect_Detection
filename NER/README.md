<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>


<!-- GETTING STARTED -->
## Getting Started

Aspect Detection as Named Entity Recognition without using any Prompt. We fine tuned [BERT](https://arxiv.org/abs/1810.04805) pretrained model for token classification.
### Prerequisites

Python == 3.8

### Installation
Install the required packages:

```
pip install -r requirements.txt
```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->
## Usage
### Train Model
There are two options for Pretrained Models: BERT-large and BERT-base. By default we used BERT-base model. You can choose model type *base* or *large* using `--model large` argument. Also we include the following arguments:

```
  --train_path, default='Data/tagged_ner_train.json
  --val_path, default='Data/tagged_ner_dev.json
  --output_dir, default = 'Models'
  --model, default = 'base'
  --epoch, default = 5
 ```
  ** Example:**
 ```
 python ner.py --model base --output_dir Models 
 
 ```
 
 ### Inference
 To evaluate our model, we request you to download pretrained models from [here]{https://drive.google.com/drive/folders/1ZK7jlUbwODJbpCS74mPiUIT6PQjcAyNv?usp=sharing}. We found best score using *Model_1* therefore we suggest it for best results. Also we used two evaluation approches: [BLEU]{https://github.com/mjpost/sacrebleu} and [BERTScore]{https://github.com/Tiiiger/bert_score}. We can choose a type *bert* or *bleu* by using `--score` argument. Also you can choose following arguments:
 
 ```
--test_path, help='Path to train data.', default='Data/tagged_ner_test.json'
--model_dir, help='Path to Pretrained model', default = './Models/Models_2.1'
--score, type = str, help='Path to result.', default='bert'
 ```
 ** Example:**
 ```
python ner_inference.py --test_path Data/tagged_ner_test.json --Models/Models_1 --score bert
 
 ```
