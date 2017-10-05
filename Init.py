
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:

data = pd.read_csv('/data/train.csv')


# In[3]:

data.head()


# In[7]:

y = data['target']
# print(y.head())
X = data.iloc[:,2:59]
# print(X.head())


# In[8]:

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:

params = {'n_estimators':[100, 250, 500, 1000, 1200, 1400, 1600], 'criterion':['gini', 'entropy'], 'max_features':['sqrt','log2',None]}
rando = RandomForestClassifier()
clf = GridSearchCV(rando, params)
clf.fit(x_train, y_train)


# In[16]:

print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)


# In[ ]:

best_model = clf.best_estimator_

y_test_preds = best_model.predict(x_test)

from pandas_ml import ConfusionMatrix
confusion_matrix = ConfusionMatrix(y_test, y_test_preds)
import matplotlib.pyplot as plt

# confusion_matrix.plot(normalized=True)
confusion_matrix.print_stats()


# In[13]:

validation = pd.read_csv('/data/test.csv')
validation.head()


# In[12]:

X_sub = validation.iloc[:,1:59]
X_sub.head()


# In[ ]:

final_predictions = best_model.predict(X_sub)
submission = pd.DataFrame({'id':validation['id'], 'target':final_predictions})
submission.head()
submission.to_csv('/output/submission.csv', sep=',', encoding='utf-8', index=False)

