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

We tried the fixed prompt LM tuning method and took aspect detection as a text generation task.
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
There are two options for Pretrained Models: BERT-large and BERT-base. By default we used BERT-base model. You can choose model type *base* or *large* using `--model` argument. Also we include the following arguments:

```
  --train_path, default='Data/tagged_ner_train.json
  --val_path, default='Data/tagged_ner_dev.json
  --output_dir, default = 'Models'
  --model, default = 'base'
  --epoch, default = 5
 ```
  **Example:**
 ```
 python ner.py --model base --output_dir Models 
 
 ```
 <p align="right">(<a href="#top">back to top</a>)</p>
 
 ### Inference
 To evaluate our model, we request you to download pretrained models from [here](https://drive.google.com/drive/folders/1ZK7jlUbwODJbpCS74mPiUIT6PQjcAyNv?usp=sharing). We found best score using *Model_1* therefore we suggest it for best results. Also we used two evaluation approches: [BLEU](https://github.com/mjpost/sacrebleu) and [BERTScore](https://github.com/Tiiiger/bert_score). One can choose a type *bert* or *bleu* by using `--score` argument. Also you can choose following arguments:
 
 ```
--test_path, Path to test, default='Data/test.jsonl'
--model_dir, Path to Pretrained model
--score, evaluation method including bert and bleu, default='bleu'
--number_prompt,the number of prompt preceding the test sentence, default=3
--model_name, The name of the LM including gpt2 and gptneo, default='gpt2'
 ```
 **Example:**
 ```
python inference_generation.py --test_path Data/test.jsonl --model_dir/Model_generation --score bert --model_name gptneo --number_prompt 1
 
 ```
 
 <p align="right">(<a href="#top">back to top</a>)</p>