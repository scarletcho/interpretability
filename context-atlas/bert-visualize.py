import json
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns

word = 'bathroom'
layer_number = 11

with open('static/jsons/' + word + '.json') as f:
    corpus = json.load(f)

df_list = []
for i in range(len(corpus['labels'])):
    sentence_set = corpus['labels'][i]['sentence']
    sentence_set = sentence_set.replace('\n','')
    sent_prev, sent_curr, sent_next = sentence_set.split(' | ')
    tag = corpus['labels'][i]['pos']
    x, y = corpus['data'][layer_number][i]
    df_list.append([word, sent_prev, sent_curr, sent_next, sentence_set, tag, x, y])

df = pd.DataFrame(df_list, columns=['word', 'sent-prev', 'sent-curr', 'sent-next', 'sentence_set', 'tag', 'x', 'y'])

plt.figure(figsize=(16,10))
p1 = sns.scatterplot(
    x="x", y="y",
    data=df,
    legend="full",
    alpha=0.9
)

texts = [plt.text(df["x"][idx], df["y"][idx], idx) for idx in range(df.shape[0])]
plt.show()

# adjust_text(texts)

pd.set_option('display.max_colwidth', -1)
print(df['sentence_set'].to_string())

