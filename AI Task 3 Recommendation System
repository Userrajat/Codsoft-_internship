import numpy as np
import pandas as pd
from tqdm import tqdm
data = pd.read_csv(r"C:\Users\Admin\Desktop\spotify.csv")
data.head()
data.info()
data.isnull().sum()
df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])
print(df)
df.corr()
from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = data.select_dtypes(include=datatypes)
print(normarization)
for col in normarization.columns:
    print(MinMaxScaler(col))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
print(kmeans)
features = kmeans.fit_predict(normarization)
print(features)
data['features'] = features
MinMaxScaler(data['features'])
songs="I Don't Wanna Be Kissed"
distance = []
song = data[(data.name.str.lower() == songs.lower())].head(1).values[0]
rec = data[data.name.str.lower() != songs.lower()]
for songs in tqdm(rec.values):
    d = 0
    for col in np.arange(len(rec.columns)):
        if not col in [1, 6, 12, 14, 18]:
            d = d + np.absolute(float(song[col]) - float(songs[col]))
    distance.append(d)
rec['distance'] = distance
rec = rec.sort_values('distance')
columns = ['artists', 'name']
print(rec[columns][:5])
