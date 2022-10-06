
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preprocessing the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import umap
import json
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

# DB_PATH = './enwiki-20170820.db'
COCA_PATH = '/Users/ycho/Music/Dropbox/5_GIT/coca-scene/results'
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

def neighbors(word, sentences, include_context=True):
  """Get the info and (umap-projected) embeddings about a word."""
  # Get part of speech of this word.
  sent_data = get_poses(word, sentences)

  # Get embeddings.
  points = get_embeddings(word.lower(), sentences, include_context)

  # Use UMAP to project down to 3 dimensions.
  points_transformed = project_umap(points)

  return {'labels': sent_data, 'data': points_transformed}

def project_umap(points):
  """Project the words (by layer) into 3 dimensions using umap."""
  points_transformed = []
  for layer in points:
    transformed = umap.UMAP().fit_transform(layer).tolist()
    points_transformed.append(transformed)
  return points_transformed

def get_embeddings(word, sentences, include_context=True):
  """Get the embedding for a word in each sentence."""
  # Tokenized input
  layers = range(-12, 0)
  points = [[] for layer in layers]
  print('Getting embeddings for %d sentences '%len(sentences))
  for sentence in sentences:
    # Apply lowercase to each sentence
    sentence = sentence.lower()

    # If include_context is set as False, only take the middle sentence (= the target sentence)
    if not include_context:
      sentence = sentence.split('|')[1]  # discarding the prev and next sentences
    else:
      tokenized_text_with_sentbound = tokenizer.tokenize(sentence)
      sentence = sentence.replace(" | ", " ")  # remove sentence boundaries ('|')

    sentence = '[CLS] ' + sentence + ' [SEP]'
    tokenized_text = tokenizer.tokenize(sentence)

    # Truncate text to match length limit of BERT encoder
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:511] + ['[SEP]']  # makes 512, after adding the last special token [SEP]

    # Find word index in tokenized sentence
    try:
      # *** Changed to lemma search from token search (Yejin)
      # To find indices of lexical variants of a word, search after lemmatization
      # (e.g., plural form of a noun, different tenses of a verb)
      lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokenized_text]

      if not include_context:
        word_idx = lemmatized_tokens.index(word)
      else:
        # To get word index matched in the target sentence (and not in the previous context sentence)
        word_idx = [i for i, x in enumerate(lemmatized_tokens)
                    if x == word and i >= tokenized_text_with_sentbound.index("|")][0]

    # If the word is made up of multiple tokens, just use the first one of the tokens that make it up.
    except:
      for i, token in enumerate(tokenized_text):
        if token == word[:len(token)]:
          word_idx = i

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # should give you something like [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    sep_idxs = [-1] + [i for i, v in enumerate(tokenized_text) if v == '[SEP]']
    segments_ids = []
    for i in range(len(sep_idxs) - 1):
      segments_ids += [i] * (sep_idxs[i+1] - sep_idxs[i])

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
      encoded_layers, _ = model(tokens_tensor, segments_tensors)
      encoded_layers = [l.cpu() for l in encoded_layers]

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    encoded_layers = [l.numpy() for l in encoded_layers]

    # Reconfigure to have an array of layer: embeddings
    for l in layers:
      try:
        sentence_embedding = encoded_layers[l][0][word_idx]
      except IndexError:
        continue
      points[l].append(sentence_embedding)

  points = np.asarray(points)
  return points

def get_sentences_COCA(word):
  """Returns a bunch of sentences from COCA-fiction"""
  print('Selecting sentences from COCA-fiction...')
  word_fpath = COCA_PATH + '/' + word + '.txt'
  with open(word_fpath) as f:
      word_sentences = f.readlines()

  print('Total number of sentences: %d'%len(word_sentences))
  np.random.shuffle(word_sentences)
  return word_sentences

def get_poses(word, sentences):
  """Get the part of speech tag for the given word in a list of sentences."""
  sent_data = []
  for sent in sentences:
    text = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in text]

    try:
      # word_idx = text.index(word)
      # Get word index in the target sentence (after a sentence boundary '|')
      # ------------------------------------------------------------------------------------
      # NB. Each mini document in COCA-fiction-scene data comprises of three sentences,
      #     which are one target sentence (that includes the query word)
      #           and two context sentences (one before and one after the target sentence):
      #
      #         "<previous sentence>|<target sentence>|<next sentence>"
      #
      # ------------------------------------------------------------------------------------
      word_idx = [i for i, x in enumerate(lemmatized) if x == word and i > text.index("|")][0]
      pos_tag = pos[word_idx][1]
    except:
      pos_tag = 'X'
    sent_data.append({
      'sentence': sent,
      'pos': pos_tag
    })

  return sent_data


if __name__ == '__main__':
  # Whether to include neighboring (prev and next) sentences as context when embedding a word
  include_context = False

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("device : ", device)

  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  # Load pre-trained model (weights)
  model = BertModel.from_pretrained('bert-base-uncased')
  model.eval()
  model = model.to(device)

  # Get selection of sentences from wikipedia.
  with open('static/words.json') as f:
    words = json.load(f)

  for word in tqdm(words):
    word_sentences = get_sentences_COCA(word)

    # Take at most 1000 sentences.
    sentences_w_word = word_sentences[:1000]

    # Process words and dump embeddings as json
    print(f'starting process for word : {word}')
    locs_and_data = neighbors(word, sentences_w_word, include_context)
    with open(f'static/jsons/{word}.json', 'w') as outfile:
      json.dump(locs_and_data, outfile)

  # Store an updated json with the filtered words.
  filtered_words = []
  for word in os.listdir('static/jsons'):
    if word.endswith('.json'):
      word = word.split('.')[0]
      filtered_words.append(word)

  with open('static/filtered_words.json', 'w') as outfile:
    json.dump(filtered_words, outfile)
  print(filtered_words)

