import numpy as np
'''import matplotlib.pyplot as plt'''
import pandas as pd
from sklearn.metrics import accuracy_score
# Importing the dataset
dataset = pd.read_csv('train dataset.csv')
X = dataset.iloc[:, [1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("\n")

print("Accuracy of KNN model: ",accuracy_score(y_test,y_pred)*100)
print("\n")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
z = classifier.predict([[21,5,4,3,6,7]])
print(z)


import keras 
from matplotlib import pyplot as plt

k_list=list(range(1,50,2))

cv_scores=[]
from sklearn.model_selection import cross_val_score

for k in k_list:
    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())

Accuracy=[x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors',fontsize=20,fontweight='bold')
plt.xlabel('Number of Neighbours K',fontsize=15)
plt.plot(k_list,Accuracy)
plt.show()
