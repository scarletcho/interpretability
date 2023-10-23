import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import KMeans
import textwrap
import re
import sys
from sklearn.metrics import pairwise_distances_argmin
pio.renderers.default = 'browser'


def get_neighbors(indices, query_id):
    query_id_mapped = dfid_to_intid[query_id]
    top10_neighbor_ids = indices[query_id_mapped][1:]  # The first nearest neighbor is the vector itself
    # return top10_neighbor_ids
    print(df.loc[query_id, 'sent'])
    print(df.loc[top10_neighbor_ids, "sent"].to_string())

# Query word and layer number to look at
# word = 'coffee'
word = sys.argv[1]
layer_number = 8

# Load json data
with open('static/jsons/' + word + '.json') as f:
    corpus = json.load(f)

# Curate a pandas dataframe
df = pd.DataFrame(corpus['labels']).T
df_xy = pd.DataFrame(corpus['data'][layer_number], columns=['x', 'y'])
df['x'] = df_xy['x'].values
df['y'] = df_xy['y'].values

kmeans = KMeans(n_clusters=12, random_state=0).fit(np.array(df_xy))

df['cluster'] = kmeans.labels_
df["cluster"] = df["cluster"].astype(str)

# Index handling
df['idx-print'] = df.index
df.index = df.index.astype(int)

# Apply some formatting to sent for better visibility
df["sent-print"] = df["sent"].apply(lambda txt: re.sub("(["+word[0].upper()+word[0].lower()+"])"+word[1:],
                                                       r"<b>\1"+word[1:]+r"</b>", txt))
df["sent-print"] = df["sent-print"].apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=40)))
fig = px.scatter(df, x="x", y="y", color="cluster", category_orders={"cluster": np.sort(df['cluster'].unique())},
                 hover_data={"x":False, "y":False, "cluster": False, "sent-print": True, "idx-print": True},
                 color_discrete_sequence=px.colors.qualitative.Light24)

fig.update_layout(
    hoverlabel=dict(
        font_size=20
    )
)

fig.update_layout(
    title={
        'text': word,
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_traces(hovertemplate="%{customdata[1]}<br>ID:%{customdata[2]}")
fig.show()
fig.write_image("./plot/" + word + ".png")
fig.write_html("./plot/" + word + ".html")

# Pandas dataframe setting + print texts
pd.set_option('display.max_colwidth', -1)  # for the full display of text
# print(df['sent'].to_string())

# Get neighbors of a specific sentence
X = np.array(df_xy)
nbrs = NearestNeighbors(n_neighbors=11).fit(X)
distances, indices = nbrs.kneighbors(X)
dfid_to_intid = dict(zip(df.index, range(len(X))))
intid_to_dfid = dict(zip(range(len(X)), df.index))
indices_mapped = [list(map(intid_to_dfid.get, indices[i])) for i in range(len(indices))]


# Inspect data

# Centroids
centroids = kmeans.cluster_centers_
nearest_to_centroids = pairwise_distances_argmin(centroids, X, metric='euclidean')
df_centroids = df.iloc[nearest_to_centroids]
# print(df_centroids['sent'])
for cent_id in range(len(df_centroids)):
    print('Centroid#:' + str(cent_id))
    cent_row = df_centroids.iloc[cent_id]
    cent_query_id = cent_row.name
    get_neighbors(indices_mapped, cent_query_id)


# query_id = 138
# get_neighbors(indices_mapped, query_id)



