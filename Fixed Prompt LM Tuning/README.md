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
There are two options for Pretrained Models: GPT-2 and GPT-Neo. You can choose model type *GPT-2* or *GPT-Neo* using `--model` argument. Also we include the following arguments:

```
  --train_path, default='Data/train.jsonl'
  --val_path, default='Data/dev.jsonl'
  --output_dir, default = 'model1'
  --model, default = 'gpt2', choices = 'gpt2' or 'gptneo'
  --epoch, default = 5
 ```
  **Example:**
 ```
 python train_generation.py --model gpt2 --output_dir model1 --epoch 3 
 
 ```
 <p align="right">(<a href="#top">back to top</a>)</p>
 
 ### Inference
To evaluate our model, we request you to download pretrained models ```generation_gpt2``` and ```generation_gptneo``` from [here]([https://drive.google.com/drive/folders/1ZK7jlUbwODJbpCS74mPiUIT6PQjcAyNv?usp=sharing](https://drive.google.com/drive/folders/1oBXWTrbu2BWYmQQmkufDSVncH2iEzlVy)). We used two evaluation approches: [BLEU](https://github.com/mjpost/sacrebleu) and [BERTScore](https://github.com/Tiiiger/bert_score). One can choose a type *bert* or *bleu* by using `--score` argument. Also you can choose following arguments:
 
 ```
--test_path, Path to test, default='Data/test.jsonl'
--model_dir, Path to Pretrained model
--score, evaluation method including bert and bleu, default='bleu'
--number_prompt,the number of prompt preceding the test sentence, default=3
--model_name, The name of the LM including gpt2 and gptneo, default='gpt2', choices ='gpt2' or 'gptneo'
 ```
 **Example:**
 ```
python inference_generation.py --test_path Data/test.jsonl --model_dir/Model_generation --score bert --model_name gptneo --number_prompt 1
 
 ```
 
 <p align="right">(<a href="#top">back to top</a>)</p>
