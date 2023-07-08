import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('diabetes.csv')
x = df.drop('Outcome',axis=1)
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

# knn_score = knn.score(x_test,y_test)

pickle.dump(knn , open('example_weights_knn.pkl','wb'))

