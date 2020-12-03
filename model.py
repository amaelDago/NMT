import torch
from torch.autograd import Variable
from torch.nn as nn
from torch.otpim as optim
import time
import pandas as pd
import numpy as np
from unicodedata import normalize
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy

#!python -m spacy download fr_core_news_sm
from sklearn.model_selection import train_test_split


def get_train_test_dataframe(filename, test_size = 0.1) : 
  english, french = [], []
  file = open(filename, encoding="utf-8", mode  ='r').readlines()
  for line in file : 
    en, fr = normalize("NFKC",line).strip().split("\t")
    english.append(en)
    french.append(fr)

  df = pd.DataFrame({"English" : pd.Series(english), "French" : pd.Series(french)})

  train, test = train_test_split(df, test_size = test_size) 
  
  return train, test  

filename = "eng-fra.txt"
test_size = 0.01
train, test = get_train_test_dataframe(filename, test_size)

# Save train and test in csv file
train.to_csv('./train.csv', index = 0)
test.to_csv("./test.csv", index = 0)

print(f"Time for writing train and test set")
time.sleep(5)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Get test and test data frame for txt file
def get_train_test_dataframe(filename, test_size = 0.1) : 
  english, french = [], []
  file = open("eng-fra.txt", encoding="utf-8", mode  ='r').readlines()
  for line in file : 
    en, fr = normalize("NFKC",line).strip().split("\t")
    english.append(en)
    french.append(fr)

  df = pd.DataFrame({"English" : pd.Series(english), "French" : pd.Series(french)})

  train, test = train_test_split(df, test_size = test_size) 
  
  return train, test 

# Load TOkenize function with spacy
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm") # If don't load, restart runtime


# Tokenize Function
def tokenize(input_str, tokenize_model) : 
  
  assert (type(input_str)==str), "Must be a string"
  input_str = re.sub(r"'re", " are", input_str)
  input_str = re.sub(r"'ve", " have", input_str)
  input_str = re.sub(r"'d", " could", input_str)
  input_str = re.sub(r"'ll", " will", input_str)
  input_str = re.sub(r"'m", " am", input_str)
  
  return [token.text for token in tokenize_model.tokenizer(input_str)]


# English tokenizer function
def enTokenizer(input_str) : 
  return tokenize(input_str, spacy_en)

# French tokenizer function
def frTokenizer(input_str) : 
  return tokenize(input_str, spacy_fr)


# English Tokenizer field
EN_TEXT = Field(
    sequential = True,
    lower = True,
    include_lengths = False,
    pad_token = "<pad>",
    unk_token = '<unk>',
    init_token = "<bos>",
    eos_token = "<eos>",
    pad_first = True,
    #batch_first  = True,
    tokenize = enTokenizer
)

# French Tokenizer field
FR_TEXT = Field(
    sequential = True,
    lower = True,
    include_lengths = False,
    init_token = "<bos>",
    eos_token = "<eos>",
    pad_first = True,
    #batch_first  = True,
    tokenize = frTokenizer
)

# Build train dataset and test dataset
train_dataset, test_dataset = TabularDataset.splits(
    format = 'csv', path = './',
    train = "train.csv", test = "test.csv",
    fields = [('English' , EN_TEXT), ('French' , FR_TEXT)] 
)

# Build vocab
EN_TEXT.build_vocab(train_dataset, test_dataset)
FR_TEXT.build_vocab(train_dataset, train_dataset)

# Train iterator
train_iter = BucketIterator(
    train_dataset,
    batch_size = 32,
    sort_key = lambda x : (len(x.English),len(x.French)),
    device = device,
    sort_within_batch = True,
    shuffle = True
)

# Test iterator
test_iter = BucketIterator(
    test_dataset,
    batch_size = 32,
    sort_key = lambda x : (len(x.English),len(x.French)),
    device = device,
    sort_within_batch = True,
    shuffle = True
)
# Show one bucket size
next(iter(test_iter))

import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
#from utils import bleu, save_checkpoint, load_checkpoint

class Transformer(nn.Module) : 
  def __init__(self, 
               embedding_size,
               src_vocab_size,
               trg_vocab_size,
               src_pad_idx,
               num_heads,
               num_encoder_layers,
               num_decoder_layers,
               forward_expansion, 
               dropout,
               max_len,
               device,
               ) : 

    super(Transformer, self).__init__()
    self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
    self.src_position_emb = nn.Embedding(max_len, embedding_size)
    self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
    self.trg_position_emb = nn.Embedding(max_len, embedding_size)
    self.device = device
    self.transformer = nn.Transformer(
       embedding_size,
       num_heads,
       num_encoder_layers,
       num_decoder_layers,
       forward_expansion,
       dropout
    )
    self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.src_pad_idx = src_pad_idx

  def make_src_mask(self, src):
    src_mask = src.transpose(0,1) == self.src_pad_idx
    return src_mask

  def forward(self, src, trg) : 
    src_seq_length, N = src.shape
    trg_seq_length, N = trg.shape

    src_positions = (
        torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
    )
    trg_positions = (
        torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
    )
  
    embed_src = self.dropout(
      (self.src_word_embedding(src) + self.src_position_emb(src_positions))
    )

    embed_trg = self.dropout(
      (self.trg_word_embedding(trg) + self.trg_position_emb(trg_positions))
    )

    src_padding_mask = self.make_src_mask(src).to(self.device)
    trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

    out = self.transformer(
        embed_src, 
        embed_trg, 
        src_key_padding_mask = src_padding_mask,
        tgt_mask = trg_mask, 
    )
    out = self.fc_out(out)
    return out
    

BOS_WORD = "<bos>"
EOS_WORD = "<eos>"

def greedy_decode_sentence(model,sentence):
    model.eval()
    sentence = enTokenizer(sentence.lower())
    indexed = [FR_TEXT.vocab.stoi[BOS_WORD]]
    for tok in sentence:
        if EN_TEXT.vocab.stoi[tok] != 0 :
            indexed.append(EN_TEXT.vocab.stoi[tok])
        else:
            indexed.append(0)
    print(indexed)
    sentence = Variable(torch.LongTensor([indexed])).to(device)
    trg_init_tok = FR_TEXT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    maxlen = 64
    for i in range(maxlen):
        pred = model(sentence.transpose(0,1), trg)
        add_word = FR_TEXT.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence+=" "+add_word
        if add_word==EOS_WORD:
            break
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).to(device)))
    return translated_sentence
    
# Set up training

# Training hyperparameters
num_epochs = 7
learning_rate = 1e-5
batch_size= 32

# Model hypermarameters
src_vocab_size = len(EN_TEXT.vocab.stoi)
trg_vocab_size = len(FR_TEXT.vocab.stoi)
embedding_size = 1024
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.10
max_len = 64
forward_expansion = 2
src_pad_idx = EN_TEXT.vocab.stoi["<pad>"]

# Initialize step
step = 0

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout, 
    max_len,
    device 
).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

pad_idx = EN_TEXT.vocab.stoi["<pad>"]

# Criterion
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

# Loop for training
for epoch in range(num_epochs) :
  start_time = time.time()
  print(f"Epoch {epoch} / {num_epochs}")

  for batch_idx, batch in enumerate(train_iter) :
    inp_data = batch.English.to(device)
    target = batch.French.to(device)

    model = model.train()
    output = model(inp_data, target[:-1])

    output = output.reshape(-1, output.shape[2])
    target = target[1:].reshape(-1)

    optimizer.zero_grad()

    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

    optimizer.step()

    step +=1
  end_time = time.time()

  # Print a translated test sentence after every epoch and running time
  print(f"Time for {epoch} epoch is {(end_time - start_time)/60} minutes")
  sentence = "I'm going at home with my friends"
  print(greedy_decode_sentence(model, sentence))