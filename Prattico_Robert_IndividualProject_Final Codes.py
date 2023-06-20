# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:05:46 2022

@author: rprat
"""

#TASK 1: CLASSIFICATION MODEL

import pandas as pd


#PART 1: DATA PREPROCESSING


#Import data
ks=pd.read_excel("C:/Users/rprat/OneDrive/Desktop/MMA Documents/INSY 662 - Visualization/Kickstarter.xlsx")

#Setting up the data
#We only want to classify a project as "successful" or "failed"
ks=ks[ks.state != "canceled"]
ks=ks[ks.state!="suspended"]

#Replace blanks in category column
ks["category"]=ks["category"].fillna("Other")

#Reset indexing
ks=ks.reset_index(drop=True)

#Drop data that we are confident is irrelevant
#This includes data captured after the launch & data that is irrelevant to project details
ks=ks.drop(["id","name", "pledged","currency","deadline","state_changed_at","created_at",
           "launched_at", "static_usd_rate","name_len", "name_len_clean","blurb_len",
           "blurb_len_clean","state_changed_at_weekday","created_at_weekday","deadline_day",
           "deadline_hr","state_changed_at_month","state_changed_at_day", "state_changed_at_yr", 
           "state_changed_at_hr", "created_at_day", "created_at_hr","launched_at_day", "launched_at_hr", 
           "launch_to_state_change_days"], axis=1)


#Data Pre-processing
x=pd.get_dummies(ks.drop(["state"],axis=1),drop_first=True)
y=ks["state"]

#Double check for any null values
x.isnull().values.any()

#Standardize with Robust Scaler
from sklearn.preprocessing import RobustScaler
rb = RobustScaler()
x_rb = rb.fit_transform(x)
x_rb=pd.DataFrame(x_rb, columns=x.columns)


#PART 2: PCA FEATURE SELECTION


#Feature selection with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=52)
pca.fit(x_rb)
variance=pca.explained_variance_ratio_

variance=list(variance)
total=0
countr=0
for i in variance: #select predictors that capture 95% of the variance
    if total<0.95:
        total+=i
        countr+=1
#The number of predictors is 2

pca = PCA(n_components=countr)
pca.fit(x_rb)
x_rb_pca = pca.transform(x_rb)


#PART 3: BUILDING THE CLASSIFICATION MODEL WITH KNN 


#Train and test sets
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain, ytest=train_test_split(x_rb_pca, y, test_size = 0.3, random_state = 0)

#Find the optimal K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
neighbors = []
cv_scores = []
for i in range (1,40,2):
    neighbors.append(i)
    knn = KNeighborsClassifier(n_neighbors=i, p=2, weights="distance")
    scores = cross_val_score(knn, xtrain, ytrain, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
#The optimal value for k is 33

#Build KNN
knn = KNeighborsClassifier(n_neighbors=optimal_k,p=2, weights="distance")
model_knn_pca = knn.fit(xtrain,ytrain)
y_test_pred = model_knn_pca.predict(xtest)

#Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy_knn_pca = accuracy_score(ytest, y_test_pred)
accuracy_knn_pca


#PART 4: EVALUATING WITH KICKSTARTER-GRADING-SAMPLE


#Setting up the data
ks_grading_sample=pd.read_excel("C:/Users/rprat/OneDrive/Desktop/MMA Documents/INSY 662 - Visualization/Kickstarter-Grading-Sample.xlsx")

#We only want to classify a project as "successful" or "failed"
ks_grading_sample=ks_grading_sample[ks_grading_sample.state != "canceled"]
ks_grading_sample=ks_grading_sample[ks_grading_sample.state!="suspended"]

#Replace blanks in category column
ks_grading_sample["category"]=ks_grading_sample["category"].fillna("Other")

#Reset indexing
ks_grading_sample=ks_grading_sample.reset_index(drop=True)

#Drop data that we are confident is irrelevant
#This includes data captured after the launch & data that is irrelevant to project details
ks_grading_sample=ks_grading_sample.drop(["id","name", "pledged","currency","deadline","state_changed_at","created_at",
                            "launched_at", "static_usd_rate","name_len", "name_len_clean","blurb_len",
                            "blurb_len_clean","state_changed_at_weekday","created_at_weekday","deadline_day",
                            "deadline_hr","state_changed_at_month","state_changed_at_day", "state_changed_at_yr", 
                            "state_changed_at_hr", "created_at_day", "created_at_hr","launched_at_day", "launched_at_hr", 
                            "launch_to_state_change_days"], axis=1)


#Data Pre-processing
x_grading_sample=pd.get_dummies(ks_grading_sample.drop(["state"],axis=1),drop_first=True)
y_grading_sample=ks_grading_sample["state"]


#Double check for any null values
x_grading_sample.isnull().values.any()

#Make Predictions
pca.fit(x_grading_sample)
x_grading_sample_pca = pca.transform(x_grading_sample)

ytestpred_grading_sample = model_knn_pca.predict(x_grading_sample_pca)
accuracy_knn_pca = accuracy_score(y_grading_sample, ytestpred_grading_sample)
accuracy_knn_pca


#PART 5: FOR GRADER: EVALUATING WITH KICKSTARTER-GRADING


#Setting up the data
ks_grading=pd.read_excel("INSERT FILE PATH")

#We only want to classify a project as "successful" or "failed"
ks_grading=ks_grading[ks_grading.state != "canceled"]
ks_grading=ks_grading[ks_grading.state!="suspended"]

#Replace blanks in category column
ks_grading["category"]=ks_grading["category"].fillna("Other")

#Reset indexing
ks_grading=ks_grading.reset_index(drop=True)

#Drop data that we are confident is irrelevant
#This includes data captured after the launch & data that is irrelevant to project details
ks_grading=ks_grading.drop(["id","name", "pledged","currency","deadline","state_changed_at","created_at",
                            "launched_at", "static_usd_rate","name_len", "name_len_clean","blurb_len",
                            "blurb_len_clean","state_changed_at_weekday","created_at_weekday","deadline_day",
                            "deadline_hr","state_changed_at_month","state_changed_at_day", "state_changed_at_yr", 
                            "state_changed_at_hr", "created_at_day", "created_at_hr","launched_at_day", "launched_at_hr", 
                            "launch_to_state_change_days"], axis=1)


#Data Pre-processing
x_grading=pd.get_dummies(ks_grading.drop(["state"],axis=1),drop_first=True)
y_grading=ks_grading["state"]


#Double check for any null values
x_grading.isnull().values.any()

#Make Predictions
pca.fit(x_grading)
x_grading_pca = pca.transform(x_grading)

ytestpred_grading = model_knn_pca.predict(x_grading_pca)
accuracy_knn_pca = accuracy_score(y_grading, ytestpred_grading)
accuracy_knn_pca




#TASK 2: CLUSTERING MODEL




import pandas as pd


#PART 1: DATA PREPROCESSING


#Import Data
#FOR GRADER: You can replace my file path with your's to test the model on the grading set
ks=pd.read_excel("C:/Users/rprat/OneDrive/Desktop/MMA Documents/INSY 662 - Visualization/Kickstarter.xlsx")
ks=ks[ks.state != "canceled"]
ks=ks[ks.state!="suspended"]
ks=ks[ks.state!="failed"]
ks=ks.reset_index(drop=True)

x=ks[["goal","launch_to_deadline_days"]]

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)


#PART 2: BUILD THE CLUSTERING MODEL WITH K-MEANS


#Find the optimal K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for i in range (2,10):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(x_std)
    labels = model.labels_
    print(i,':',silhouette_score(x_std,labels))

#Cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(x_std)
classify = model.predict(x_std)

#Visualize
from matplotlib import pyplot
plot=pyplot.scatter(ks['goal'], ks['launch_to_deadline_days'],  c=model.labels_, cmap='rainbow') 
    
#Evaluate clustering solution
silhouette_score(x_std, labels)


#PART 3: INFERRING INSIGHTS 


#Defining Cluster sizes
cluster_map = pd.DataFrame()
cluster_map['data_index'] = ks.index.values
cluster_map['cluster'] = model.labels_
cluster_map

cluster0=cluster_map[cluster_map.cluster == 0]
cluster1=cluster_map[cluster_map.cluster == 1]
cluster2=cluster_map[cluster_map.cluster == 2]
cluster3=cluster_map[cluster_map.cluster == 3]

#Cluster 0
cluster0dataindex=cluster0["data_index"].tolist()
cluster0goaldata=[]
cluster0ltddata=[]
for i in cluster0dataindex:
    cluster0goaldata.append(x["goal"].loc[i])
for i in cluster0dataindex:
    cluster0ltddata.append(x["launch_to_deadline_days"].loc[i])
min(cluster0goaldata)
min(cluster0ltddata)
max(cluster0goaldata)
max(cluster0ltddata)

#Cluster 1
cluster1dataindex=cluster1["data_index"].tolist()
cluster1goaldata=[]
cluster1ltddata=[]
for i in cluster1dataindex:
    cluster1goaldata.append(x["goal"].loc[i])
for i in cluster1dataindex:
    cluster1ltddata.append(x["launch_to_deadline_days"].loc[i])
min(cluster1goaldata)
min(cluster1ltddata)
max(cluster1goaldata)
max(cluster1ltddata)

#Cluster 2
cluster2dataindex=cluster2["data_index"].tolist()
cluster2goaldata=[]
cluster2ltddata=[]
for i in cluster2dataindex:
    cluster2goaldata.append(x["goal"].loc[i])
for i in cluster2dataindex:
    cluster2ltddata.append(x["launch_to_deadline_days"].loc[i])
min(cluster2goaldata)
min(cluster2ltddata)
max(cluster2goaldata)
max(cluster2ltddata)

#Cluster 3
cluster3dataindex=cluster3["data_index"].tolist()
cluster3goaldata=[]
cluster3ltddata=[]
for i in cluster3dataindex:
    cluster3goaldata.append(x["goal"].loc[i])
for i in cluster3dataindex:
    cluster3ltddata.append(x["launch_to_deadline_days"].loc[i])
min(cluster3goaldata)
min(cluster3ltddata)
max(cluster3goaldata)
max(cluster3ltddata)