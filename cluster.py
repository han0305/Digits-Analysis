
"""
@author: han0305
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print('training data {}'.format(x_train.shape))
print('training labels {}'.format(y_train.shape))

print('test data {}'.format(x_test.shape))
print('test labels {}'.format(y_test.shape))

fig,axs = plt.subplots(3,3,figsize = (12,12))
plt.gray()

for i,ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(y_train[i]))
    
fig.show()


x = x_train.reshape(len(x_train),-1)
y = y_train
print(x.shape)

x = x.astype(float)/255.

from sklearn.cluster import MiniBatchKMeans

n_digits = len(np.unique(y_test))
print(n_digits)

kmeans = MiniBatchKMeans(n_clusters = n_digits)
kmeans.fit(x)
print(kmeans.labels_[:20])

def infer_cluster_labels(kmeans,actual_labels):
    inferred_labels={}
    for i in range(kmeans.n_clusters):
        labels=[]
        index = np.where(kmeans.labels_==i)
        labels.append(actual_labels[index])
        if len(labels[0]) == 1:
            counts=np.bincount(labels[0])
        else:
            counts=np.bincount(np.squeeze(labels))
        if np.argmax(counts) in inferred_labels:
            inferred_labels[np.argmax(counts)].append(i)
        else:
            inferred_labels[np.argmax(counts)] = [i]
    return inferred_labels

def infer_data_labels(x_labels, cluster_labels):
    predicted_labels = np.zeros(len(x_labels)).astype(np.uint8)
    
    for i,cluster in enumerate(x_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i]=key
    return predicted_labels
    
    
cluster_labels = infer_cluster_labels(kmeans,y)
x_clusters = kmeans.predict(x)
predicted_labels = infer_data_labels(x_clusters,cluster_labels)
print(predicted_labels[:20])
print(y[:20])  

from sklearn import metrics

def calculate_metrics(estimator, data, labels):

    
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))
        
clusters = [10, 16, 36, 64, 144, 256]


for n_clusters in clusters:
    estimator = MiniBatchKMeans(n_clusters = n_clusters)
    estimator.fit(x)
    
    
    calculate_metrics(estimator, x, y)
    
    
    cluster_labels = infer_cluster_labels(estimator, y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
    
    
    print('Accuracy: {}\n'.format(metrics.accuracy_score(y, predicted_Y)))            
            

x_test = x_test.reshape(len(x_test),-1)


x_test = x_test.astype(float) / 255.


kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(x)
cluster_labels = infer_cluster_labels(kmeans, y)


test_clusters = kmeans.predict(x_test)
predicted_labels = infer_data_labels(kmeans.predict(x_test), cluster_labels)
    

print('Accuracy: {}\n'.format(metrics.accuracy_score(y_test, predicted_labels)))           



kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(x)


centroids = kmeans.cluster_centers_


images = centroids.reshape(36, 28, 28)
images *= 255
images = images.astype(np.uint8)


cluster_labels = infer_cluster_labels(kmeans, y)


fig, axs = plt.subplots(6, 6, figsize = (20, 20))
plt.gray()


for i, ax in enumerate(axs.flat):
    
    
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))
    

    ax.matshow(images[i])
    ax.axis('off')
    

fig.show()




