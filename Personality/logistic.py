import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pickle

traindata =pd.read_csv('train dataset.csv')
array = traindata.values

df=pd.DataFrame(array)
maindf =df[[1,2,3,4,5,6]]
mainarray=maindf.values
#print(mainarray)

temp=df[7]
train_y =temp.values
# print(train_y)
# print(mainarray)

for i in range(len(train_y)):
	train_y[i] =str(train_y[i])

classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
classifier.fit(mainarray, train_y)

with open('classifier.pk1', 'wb') as pickle_file:
        pickle.dump(classifier , pickle_file)

with open('classifier.pk1', 'rb') as pickle_file:
        classifierpk = pickle.load(pickle_file)
       
y_pred = classifierpk.predict([[18,7,6,3,7,1]])
print(y_pred)
