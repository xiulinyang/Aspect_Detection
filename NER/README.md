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
##Usage
### Train Model
There are two options for Pretrained Models: BERT-large and BERT-base. By default we used BERT-base model. You can choose model type base or large using `--model large` argument. Also we include the following arguments:

```
  --train_path, default='Data/tagged_ner_train.json
  --val_path, default='Data/tagged_ner_dev.json
  --output_dir, default = 'Models'
  --model, default = 'base'
  --epoch, default = 5
 ```
 Example: python ner.py --model base --output_dir Models 
 
 ###Inference
 
