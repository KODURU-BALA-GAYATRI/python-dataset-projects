import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

data=pd.read_csv("EM.csv")
print("input data and shape")
print(data.shape)
print(data.head())

f1=data['v1'].values
f2=data['v2'].values
x=np.array(list(zip(f1,f2)))

print("x",x)
print("graph for whole dataset")
plt.scatter(f1,f2,c="black",s=7)
plt.show()

Kmeans=KMeans(20,random_state=0)
labels=Kmeans.fit(x).predict(x)
print("labels",labels)

centroids=Kmeans.cluster_centers_
print("centroid ds",centroids[:5])
plt.scatter(x[:,0],x[:,1],c=labels,s=40,cmap='viridis');

print("graph using Kmeans algorithm")
plt.scatter(centroids[:,0],centroids[:,1],marker="*",s=200,c="#050505");
plt.show()

gmm=GaussianMixture(n_components=3).fit(x)
label=gmm.predict(x)
probs=gmm.predict_proba(x)
size=10*probs.max(1)**3
print("graph using em algorithm")
plt.scatter(x[:,0],x[:,1],c=labels,s=size,cmap="viridis");
plt.show()