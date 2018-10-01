
# coding: utf-8

# In[11]:

import pandas as pd
import numpy as np
import pickle
from sklearn.base import TransformerMixin
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


# In[12]:

# relative path to current Python work path 
modelFile = 'dtree_model_obj.pkl'
scoreInputCSV = 'hmeq.csv'
scoreOutputCSV = 'scoreOut.csv'


# In[13]:

inputDf = pd.read_csv(scoreInputCSV)

targetVars = ['BAD']
inVars = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

model = open(modelFile, 'rb')
dtree = pickle.load(model)
model.close()

tmpDf = DataFrameImputer().fit_transform(inputDf[inVars])
outputDf = pd.DataFrame(dtree.predict_proba(tmpDf))


# In[14]:

outputcols = map(lambda x:'P_BAD' + str(x) ,list(dtree.classes_))
outputDf.columns = outputcols
outputDf = pd.merge(inputDf,outputDf,how='inner',left_index=True,right_index=True)


# In[15]:

print(outputDf.tail())


# In[16]:

outputDf.to_csv(scoreOutputCSV,sep=',',index=False)


# In[ ]:



