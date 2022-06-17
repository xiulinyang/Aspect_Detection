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

Tuning-Free prompt learning using GPT-2 and GPT-neo as Language models with custom templates.
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
There are two options for Pretrained Models: GPT-2 and GPT-neo. You can choose model type *base* or *large* using `--model` argument. Also we include the following arguments:

```
  --model_name, default="neo"
  --train_path, default="data/train.jsonl"
  --test_path, default="data/test.jsonl"
  --n_gen, default=15,
 ```
  **Example:**
 ```
python tuning_free_prompt.py --model_name "neo" --train_path "data/train.jsonl" --test_path "data/test.jsonl" --n_gen 20
 
 ```
 
 <p align="right">(<a href="#top">back to top</a>)</p>
