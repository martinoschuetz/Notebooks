
# coding: utf-8

# In[26]:

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


# In[27]:

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[28]:

# load data
df = pd.read_csv("hmeq.csv",na_values=['NULL'])
targetvar = ['BAD']
inputvars = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
Y = df[targetvar]
X0 = df[inputvars]
# impute the missing value
X = DataFrameImputer().fit_transform(X0)


# In[29]:

# split the data into two parts - train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)


# In[30]:

# build a decision tree model
dt = DecisionTreeClassifier(random_state=99)
mdl=dt.fit(X_train, Y_train)
mdl.predict_proba(X_test)
print("accuracy on training set: %f" % mdl.score(X_train, Y_train))
print("accuracy on test set: %f" % mdl.score(X_test, Y_test))


# In[31]:

#export the model to a external file for later use
import pickle 
pklfile = open("dtree_model_obj.pkl",'wb')
pickle.dump(mdl, pklfile)
pklfile.close()


# In[ ]:



