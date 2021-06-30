import pickle


Open = 4
Cons = 2
Extra = 6
Aggre = 6
Neuro = 4

with open('classifier.pk1', 'rb') as pickle_file:
        classifierpk = pickle.load(pickle_file)

y_pred = classifierpk.predict([[18,Open,Cons,Extra,Aggre,Neuro]])
#y_pred = classifierpk.predict([[18,7,6,3,7,1]])
print(y_pred)
