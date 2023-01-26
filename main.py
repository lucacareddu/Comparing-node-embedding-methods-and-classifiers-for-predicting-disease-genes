import numpy as np
import pandas as pd
import random as rd

import networkx as nx
import time
from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps, HOPE, GLEE, DeepWalk, Node2Vec

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from my_cnn import MY_CNN

from sklearn.metrics import accuracy_score #ratio of correctly predicted samples
from sklearn.metrics import precision_score #tp/(tp+fp)
from sklearn.metrics import recall_score #tp/(tp+fn)
from sklearn.metrics import average_precision_score 
# divides recall and accuracy in threshold based intervals and compute area (sums) of R(n)-R(n-1),P(n) curve
from sklearn.metrics import roc_auc_score
# area of the curve true positive rate (recall), false positive rate ( fp/(fp+tn) )
from sklearn.metrics import f1_score # 2 * (precision * recall) / (precision + recall)
from sklearn.metrics import matthews_corrcoef 
# correlation coefficient in [-1,1] (best=1, 0=random, -1=inverse) that take into account true/false positive/negative predictions

#############MODELS#############

embedders = [
            # LaplacianEigenmaps(dimensions=64), 
            # HOPE(dimensions=64),
            GLEE(dimensions=127), #this embedder returns embeddings dimensions+1 long
            # DeepWalk(walk_number = 20, walk_length = 50, dimensions = 64, workers = -1),
            # Node2Vec(walk_number = 5, walk_length = 10, dimensions = 64, workers = -1)
            ]

classifiers = [
            # RandomForestClassifier(n_estimators=200, n_jobs=-1),
            # MLPClassifier(hidden_layer_sizes=(256)), # 128 = len(dis_emb|gen_emb),
            MY_CNN(epochs=10)
            ]

################################

dataset='dataset.csv'
test_size=0.2

################################

def get_data(dataset=dataset):
    df=pd.read_csv(dataset, delimiter='\t')
    df=df.drop("Disease Name",axis=1)
    df["# Disease ID"] = '#' + df["# Disease ID"] #used to distinguish diseases column from genes one
    g = nx.from_pandas_edgelist(df,source="Gene ID",target="# Disease ID")
    g = nx.convert_node_labels_to_integers(g, label_attribute="revert")
    return g,nx.get_node_attributes(g,"revert"),np.triu(nx.adjacency_matrix(g).todense())
    #revert attribute is a dictionary that allow to reconstruct the original graph
    #adj matrix is symmetric, so to avoid double counting edges I transform it into an upper triangular matrix via np.triu

def get_embeddings(g):
    print("Generating embeddings..")
    for m in embedders:
        print(m.__class__.__name__, end=' ')
        t0 = time.time()
        m.fit(g)
        print("done. %ss" % (time.time()-t0))
    return [m.get_embedding() for m in embedders]

def get_sample(embedding,i,j,first_is_dis):
    to_stack=(embedding[i],embedding[j]) if first_is_dis else (embedding[j],embedding[i])
    return np.vstack(to_stack) #horizontally (vertically for the CNN) stacked embeddings (ordered) -> dis_emb|gen_emb

def get_zero_indexes(adj_mat,num_ones):
    zero_rows,zero_cols=np.where(adj_mat==0)
    random_zero_indexes=rd.sample(range(len(zero_rows)), num_ones) #pick n=num_ones random indices to have len(pos_samples)==len(neg_samples) and so achieve a balanced training
    return zero_rows[random_zero_indexes],zero_cols[random_zero_indexes]    

def get_stuff(embedding,nodes_dict,adj_mat):
    #creating a matrix of only positive samples
    one_rows,one_cols=np.where(adj_mat==1)
    pos_samples=np.array([get_sample(embedding,i,j,'#' in str(nodes_dict.get(i))) for i,j in zip(one_cols,one_rows)])
    
    #creating a matrix of only negative samples
    zero_rows,zero_cols = get_zero_indexes(adj_mat,len(one_rows))
    neg_samples=np.array([get_sample(embedding,i,j,'#' in str(nodes_dict.get(i))) for i,j in zip(zero_rows,zero_cols)])
    
    #creating a matrix of the mixed and shuffled samples along with its labels matrix
    samples,labels=shuffle(np.vstack((pos_samples,neg_samples)),np.hstack((np.ones(len(pos_samples)),np.zeros(len(neg_samples)))))
    
    return train_test_split(samples, labels, test_size=test_size)

def get_metrics(labels,predictions):
    metrics_names=["ACC","PREC","REC","APREC","ROC_AUC","F1","MCC"]
    metrics=[accuracy_score(labels, predictions),
            precision_score(labels, predictions),
            recall_score(labels, predictions),
            average_precision_score(labels, predictions),
            roc_auc_score(labels, predictions),
            f1_score(labels, predictions),
            matthews_corrcoef(labels, predictions)
            ]
    return dict(zip(metrics_names, metrics))

def train_and_predict(samples_train,labels_train,samples_test,labels_test):
    for m in classifiers:
        print(m.__class__.__name__," training..", end=' ')
        t0 = time.time()
        m.fit(samples_train,labels_train) 
        print("done. %ss" % (time.time()-t0))
        print("Predicting test set..", end=' ')
        predictions=np.rint(m.predict(samples_test)) #np.rint round reals to integers
        print("done.")
        scores=get_metrics(labels_test,predictions)
        print()
        for k,v in scores.items(): print(k,"-> ",v)

if __name__=="__main__":
    g,nodes_dict,adj_mat=get_data()

    embeddings=get_embeddings(g)
    
    for i in range(len(embeddings)):
        print("\nEvaluating ",embedders[i].__class__.__name__," embeddings:")

        samples_train,samples_test,labels_train,labels_test = get_stuff(embeddings[i],nodes_dict,adj_mat)
        
        train_and_predict(samples_train,labels_train,samples_test,labels_test)