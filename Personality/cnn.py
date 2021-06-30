
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

traindata =pd.read_csv('train dataset.csv')
array = traindata.values


for i in range(len(array)):
	if array[i][0]=="Male":
		array[i][0]=1
	else:
		array[i][0]=0


x=array[:,:7]
print(x)
y=array[:,7]
print(y)

y_=y.reshape(-1,1)

encoder=OneHotEncoder(sparse=False)
y=encoder.fit_transform(y_)


testdata=pd.read_csv('test dataset.csv')
array1=testdata.values

for i in range(len(array1)):
	if array1[i][0]=="Male":
		array1[i][0]=1
	else:
		array1[i][0]=0

x1=array1[:,:7]
print(x1)
y1=array1[:,7]
print(y1)

y_1=y1.reshape(-1,1)

encoder=OneHotEncoder(sparse=False)
y1=encoder.fit_transform(y_1)


def deepml_model():
	deepml=Sequential()
	deepml.add(Dense(10,input_dim=7,activation='relu'))
	deepml.add(Dense(10,activation='relu'))
	deepml.add(Dense(5,activation='softmax'))
	deepml.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return deepml


estimate=KerasClassifier(build_fn=deepml_model,epochs=300,batch_size=7,verbose=0)

k_fold=KFold(n_splits=5,shuffle=True,random_state=7)

results=cross_val_score(estimate,x1,y1,cv=k_fold)

print("Model:%.2f%% (%.2f%%)" %(results.mean()*100,results.std()*100))

'''
import keras
from matplotlib import pyplot as plt
model1=deepml_model()
history = model1.fit(x1, y1,validation_split = 0.3, epochs=300, batch_size=7)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#accuracy=70%
'''





