import torch
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel

from embedding import PROMPTEmbedding

from sacrebleu.metrics import BLEU
from bert_score import score

import pandas as pd

def getInputs(prompt, word=None):
    if word:
        prompt += word

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs['input_ids'] = torch.cat([torch.full((1, NTOKENS), 50256), inputs['input_ids']], 1).to(DEVICE)
    inputs['attention_mask'] = torch.cat([torch.full((1, NTOKENS), 1), inputs['attention_mask']], 1).to(DEVICE)

    return inputs


def getDecoded(outputs):
    tokout = outputs.logits[0][-1].argmax()
    return tokenizer.decode(tokout, skip_special_tokens=True)


def template(sentence, aspect=""):
    return f"Sentence:{sentence}. Aspect:{aspect}"


def preprocessText(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = text.replace("  ", " ")
    return text


def promptLoader(test_dataset):
    test_dt = []
    test_aspect = []
    test_sentence = []
    for i, sent in enumerate(test_dataset['sentence']):
        train_idxs = np.random.randint(0, len(test_dataset) - 1, 3)
        text = ""
        for ti in train_idxs:
            sentence = test_dataset.iloc[ti]["sentence"]
            sentence = preprocessText(sentence)

            aspectlist = [preprocessText(txt) for txt in test_dataset.iloc[ti]["aspect_pos_string"]]
            aspects = ", ".join(aspectlist) + "."

            text += template(sentence, aspects)

        sentence = test_dataset.iloc[i]["sentence"]
        sentence = preprocessText(sentence)
        test_sentence.append(sentence)
        text += template(sentence)
        test_dt.append(text)
        tst_aspect = ', '.join(test_dataset.iloc[i]['aspect_pos_string'])
        test_aspect.append(tst_aspect)

    return test_dt, test_aspect, test_sentence


def cleanAspect(final_sentect, prompt):
    aspect = final_sentence[len(prompt):].split(".")[0]
    for key in SPECIAL_KEYS:
        aspect = aspect.replace(key, "")

    return aspect

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
    scores = bleu.corpus_score(pred,test)
    return scores

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Options: 'neo' for GPT-neo & 'gpt2' for GPT-2")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for testing")
    parser.add_argument("--test_path", type=str, help="path to test jsonl file", default="data/test.jsonl")
    parser.add_argument("--n_gen", type=int, help="number of text generations", default=15)

    args = parser.parse_args()

    NTOKENS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                      "eos_token": "<|EOS|>",
                      "unk_token": "<|UNK|>",
                      "pad_token": "<|PAD|>",
                      "sep_token": "<|SEP|>"}
    SPECIAL_KEYS = list(SPECIAL_TOKENS.values())

    if args.model_name == "neo":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    elif args.model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("Model and tokenizer initialized....")

    model.resize_token_embeddings(len(tokenizer))
    print("Token embeddings resized...")


    for p in model.parameters():
        p.requires_grad = False

    prompt_emb = PROMPTEmbedding(model.get_input_embeddings(),
                                 n_tokens=NTOKENS,
                                 initialize_from_vocab=True)

    model.set_input_embeddings(prompt_emb)
    print("Set Input embeddings...")

    model.to(DEVICE)

    ckp = torch.load(args.checkpoint)
    model.load_state_dict(ckp["model_state_dict"])

    test_df = pd.read_json(args.test_path, lines=True)
    test_dt, test_apsect, test_sentence = promptLoader(test_df)
    print("Test dataset loaded...")

    n_gen = args.n_gen

    pred_aspects = []

    print("Begin Testing....")
    for prompt, label in tqdm(zip(test_dt, test_apsect)):
        final_sentence = prompt
        word = None
        for i in range(n_gen):
            inputs = getInputs(prompt, word)
            outputs = model(**inputs)
            word = getDecoded(outputs)
            final_sentence += word

        aspect = cleanAspect(final_sentence, prompt)
        pred_aspects.append(aspect)

    bleu_score = get_bleu_score(pred_aspects, test_apsect)
    bert_scores = get_bert_score(pred_aspects, test_apsect)
    print(bert_scores)
