import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import numpy as np

def get_neighbors(indices, query_sent_id):
    top10_neighbor_ids = indices[query_sent_id][1:]  # The first nearest neighbor is the vector itself
    print('< Query sentence >\n', df.loc[query_sent_id, "sentence_set"].replace(" | ", "\n "), '\n')
    print(df.loc[top10_neighbor_ids, "sentence_set"].to_string().replace(" | ", "\n\t\t"))


word = 'bathroom'
layer_number = 11

with open('static/jsons/' + word + '.json') as f:
    corpus = json.load(f)

df_list = []
for i in range(len(corpus['labels'])):
    sentence_set = corpus['labels'][i]['sentence']
    sentence_set = sentence_set.replace('\n', '')
    sent_prev, sent_curr, sent_next = sentence_set.split(' | ')
    tag = corpus['labels'][i]['pos']
    x, y = corpus['data'][layer_number][i]
    df_list.append([word, sent_prev, sent_curr, sent_next, sentence_set, tag, x, y])

df = pd.DataFrame(df_list, columns=['word', 'sent-prev', 'sent-curr', 'sent-next', 'sentence_set', 'tag', 'x', 'y'])

# plt.figure(figsize=(16, 10))
plt.figure(figsize=(40, 40))
p1 = sns.scatterplot(x="x", y="y", data=df, legend="full", alpha=0.9)

texts = [plt.text(df["x"][idx], df["y"][idx], idx) for idx in range(df.shape[0])]
plt.show()

pd.set_option('display.max_colwidth', -1)
print(df['sentence_set'].to_string())

# Get neighbors of a specific sentence
X = np.array(df[['x', 'y']].values.tolist())
nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

query_sent_id = 141
get_neighbors(indices, query_sent_id)

