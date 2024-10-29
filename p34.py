import numpy as np
data=np.array([[18,1,1,250,0],[20,0,1,150,0],[22,0,2,150,1],[23,1,2,250,1],
[24,1,3,150,0],[24,0,2,250,0],[24,0,3,150,1],[25,0,2,250,0],
[25,0,1,150,0],[25,1,3,150,1],[25,1,5,250,1],[25,0,5,250,1],
[26,0,1,150,0],[26,0,2,250,0],[26,0,3,150,1],[27,1,3,250,1],
[27,1,5,250,1],[27,1,7,250,1],[27,0,5,150,1],[27,0,7,250,1],
[27,0,7,150,1],[28,1,3,250,0],[28,1,5,250,1],[28,1,7,250,1],
[28,1,10,250,1],[29,0,10,250,1],[30,1,5,150,1],[30,0,7,250,1]])
print(data)

x=data[:,:-1]
y=data[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

import pandas as pd
df=pd.DataFrame({"ytest":ytest,"ypred":ypred})
print(df)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))

print("predict=",model.predict([[26,1,5,250]]))
