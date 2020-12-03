import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import spacy
import json
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""


"""
!python -m spacy download en_core_web_sm
!python -m spacy download fr_core_web_sm
"""

# Get device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Load spacy model for english and french
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm") # If don't load, restart runtime

# Load english vocabulary
with open('EN_VOCAB.json', 'r') as fp:
    EN_VOCAB = json.load(fp)

# Load french vocabulary
with open('FR_VOCAB.json', 'r') as fp:
    FR_VOCAB = json.load(fp)
#print(f"Lengths of english vocab is : {len(FR_VOCAB)}, and french ones is {len(EN_VOCAB)}")

def tokenize(input_str, tokenize_model) : 
  
  assert (type(input_str)==str), "Must be a string"
  input_str = re.sub(r"'re", " are", input_str.lower())
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
  
 
# Define Transformer
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
    model.eval().to(device)
    sentence = enTokenizer(sentence.lower())
    indexed = [EN_VOCAB[BOS_WORD]]
    for tok in sentence:
        if tok in EN_VOCAB :
            indexed.append(EN_VOCAB[tok])
        else:
            indexed.append(0)
    print(indexed)
    sentence = Variable(torch.LongTensor([indexed])).to(device)
    trg_init_tok = EN_VOCAB[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    maxlen = 64
    for i in range(maxlen):
        pred = model(sentence.transpose(0,1), trg)
        add_word = list(FR_VOCAB.keys())[pred.argmax(dim=2)[-1]]
        translated_sentence+=" "+add_word
        if add_word==EOS_WORD:
            break
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).to(device)))
    return translated_sentence