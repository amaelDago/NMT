import numpy as np
import os
import random
import json
from flask import Flask, render_template, request
from utils import greedy_decode_sentence
import torch
from utils import Transformer, greedy_decode_sentence

os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get device
device = torch.device('cpu')

# Load english vocabulary
with open('EN_VOCAB.json', 'r') as fp:
    EN_VOCAB = json.load(fp)

# Load french vocabulary
with open('FR_VOCAB.json', 'r') as fp:
    FR_VOCAB = json.load(fp)

# Model hypermarameters
src_vocab_size = len(EN_VOCAB)
trg_vocab_size = len(EN_VOCAB)
embedding_size = 1024
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.10
max_len = 64
forward_expansion = 4
src_pad_idx = EN_VOCAB["<pad>"]

# Instanciate model
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

# Load model
#model = torch.load("model.pkl")
model = torch.load("model.pkl", map_location = "cpu")
model.eval()

print(f"Model as load")

#unknown = "__unk__"


# Define application
app = Flask(__name__)

# Deine route for homepage
@app.route("/")

def index() : 
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict() : 

	sentence = [str(x) for x in request.form.values()]
	sentence = sentence[0]
	print(sentence)
	if not sentence.endswith('.') : 
		sentence += '.'
	translated_sentence = greedy_decode_sentence(model, sentence)
	return render_template('index.html', prediction_text= 'The translation of \"{}\" is \n '.format(sentence[0], translated_sentence))

if __name__ == "__main__" : 
    app.run(debug = True)
	