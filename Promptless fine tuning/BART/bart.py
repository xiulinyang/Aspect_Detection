import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import nltk



path = '/NER/Data/train.csv'
df = pd.read_csv(path)
df = df.dropna().reset_index()

df = df[['aspect_pos_string','sentence']]

#Clean text
df['sentence'] = df['sentence'].apply(lambda x: x.replace('\n',''))

#Call model
pretrained_model = "sshleifer/distilbart-cnn-6-6"
hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model,
model_cls=BartForConditionalGeneration)

#Tokenization
hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)

#Preprocessing and setting parameters
preprocessor = SummarizationPreprocessor(
    hf_tokenizer,
    id_attr="index",
    text_attr="sentence",
    target_text_attr = "aspect_pos_string",
    
    max_input_tok_length =156,
    max_target_tok_length = 150,
    min_summary_char_length = 30,
       
)

#Hyper_parameters and fine-tuning
proc_df = preprocessor.process_df(aspects)

text_gen_kwargs = default_text_gen_kwargs(hf_config, hf_model, task="summarization")
batch_tokenize_transform = Seq2SeqBatchTokenizeTransform(
    hf_arch, hf_config, hf_tokenizer, hf_model, text_gen_kwargs=
 {'max_length': 20,'min_length': 3,'do_sample': False, 'early_stopping': True, 'num_beams': 4, 'temperature': 1.0, 
  'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
 'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
 'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
 'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False}
)

blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=batch_tokenize_transform),noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader("sentence"), get_y=ColReader("aspect_pos_string"), splitter=RandomSplitter())

#splitter=GrandparentSplitter(train_name='/content/sample_data/train.csv', valid_name='/content/sample_data/test.csv')

#batch size and dataloader
dls = dblock.dataloaders(proc_df, bs=2)
one=dls.one_batch()
len(one), one[0]["input_ids"].shape, one[1].shape

#Metrics
seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'fr' },
            'returns': ["precision", "recall", "f1"]}}

#Model
model = BaseModelWrapper(hf_model)
learn_cbs = [BaseModelCallback]
fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

#Training process
learn = Learner(dls, model,
                opt_func=ranger,loss_func=PreCalculatedCrossEntropyLoss(),
                cbs=learn_cbs,splitter=partial(blurr_seq2seq_splitter, arch=hf_arch)).to_fp32()

#optimizer
learn.create_opt() 
learn.freeze()

#One batch
one_batch_only = dls.one_batch()

#predictions
prediction = learn.model(one[0])
prediction
len(prediction), prediction["loss"].shape,prediction["logits"].shape

#find optimal lr
learn.lr_find(suggest_funcs=[minimum,steep,valley,slide])

#number of epochs and learning rate set up
learn.fit_one_cycle(3, lr_max=4e-4,cbs=fit_cbs)

#nltk.download('punkt')
#Results and length of sentence and aspect 
learn.show_results(max_n=9, learner=learn,input_trunc_at=750, target_trunc_at=175)
