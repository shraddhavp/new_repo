
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

traindata =pd.read_csv('train dataset.csv')
array = traindata.values

for i in range(len(array)):
    if array[i][0]=="Male":
        array[i][0]=1
    else:
        array[i][0]=0


df=pd.DataFrame(array)

maindf =df[[0,1,2,3,4,5,6]]
mainarray=maindf.values
print (mainarray)


temp=df[7]
train_y =temp.values
# print(train_y)
# print(mainarray)

for i in range(len(train_y)):
    train_y[i] =str(train_y[i])


mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
history=mul_lr.fit(mainarray, train_y)
'''
with open('classifier.pk1', 'wb') as pickle_file:
    pickle.dump(mul_lr , pickle_file)
    '''
    
testdata =pd.read_csv('test dataset.csv')
test = testdata.values

for i in range(len(test)):
    if test[i][0]=="Male":
        test[i][0]=1
    else:
        test[i][0]=0


df1=pd.DataFrame(test)

testdf =df1[[0,1,2,3,4,5,6]]
maintestarray=testdf.values
print(maintestarray)

y_pred = mul_lr.predict(maintestarray)
for i in range(len(y_pred)) :
    y_pred[i]=str((y_pred[i]))
#DF = pd.DataFrame(y_pred,columns=['Predicted Personality'])
#DF.index=DF.index+1
#DF.index.names = ['Person No']
#DF.to_csv("output1.csv")
#print(y_pred)
#print(test[:, 7])
z = mul_lr.predict([[21,5,4,3,6,7,5]])
print(z)

print("Accuracy is of LogisticRegression is : ",(accuracy_score(test[:,7],y_pred))*100)

#got an accuracy of 85.75%


#to find area under roc curve
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
auc1 = multiclass_roc_auc_score(test[:,7], y_pred, average="macro")
print("Area under curve : ", auc1)


#to plot roc curve
import matplotlib.pyplot as plt
#y_test=test[:,7]

def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc1 = dict()
 
    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc1[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc1[i], i+1))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    #sns.despine()
    plt.show()

plot_multiclass_roc(mul_lr, test[:,:7], test[:,7], n_classes=5, figsize=(16, 10))

