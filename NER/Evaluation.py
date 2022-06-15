from sacrebleu.metrics import BLEU
import bert_score
from bert_score import score

def get_bleu_score(pred, test):
    '''
    This function calculate the blue score for the prediction of the model
    Argument:
    pred: the prediction of the model which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    test: the ground truth of the target labels which is a list of strings e.g., ['asp1, asp2', 'asp1, asp2, asp3']
    Output: Bleu Score Bleu Score for unigram/Bleu Score for bigram/Bleu Score for trigram
    '''
    print(pred, test)
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

