
# coding: utf-8

# In[128]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve
import numba


# In[3]:


# from matplotlib import pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[57]:


churnData = pd.read_csv('http://www.dataminingconsultant.com/data/churn.txt')


# In[58]:


churnData['Churn'] = churnData['Churn?'].map(lambda x: 1 if x=='True.' else 0)


# In[59]:


# remove columns that we won't use 
churnData1 = churnData.drop(['Area Code','Phone','State','Churn?'],axis=1)


# In[60]:


# find out which columns are object type and convert them to dummy vars
list_ = list(churnData1.select_dtypes(include=['object']).columns)
churnData2 = pd.get_dummies(churnData1, prefix=list_)


# In[61]:


churnData2.head(4)


# In[62]:


# merge with the other columns
list_nonObj = list(churnData.select_dtypes(exclude=['object']).columns)
churnData3 = churnData[list_nonObj]


# In[63]:


inputData = pd.concat([churnData2,churnData3], axis = 1)


# In[64]:


inputData.head()


# In[65]:


# split the input data set into train and test 
churnData_train, churnData_test = train_test_split(inputData, test_size =0.3)


# In[66]:


features = inputData.drop(['Churn'], axis=1).columns


# In[67]:


rcClassifier = RandomForestClassifier(n_estimators=40)
rcClassifier.fit(churnData_train[features], churnData_train['Churn'])


# In[69]:


# Make Predictions
predictions = rcClassifier.predict(churnData_test[features])
probabilities = rcClassifier.predict_proba(churnData_test[features])


# In[70]:


score = rcClassifier.score(churnData_test[features],churnData_test['Churn'])
print "Accuracy : ", score


# In[76]:


featImport = zip(features, rcClassifier.feature_importances_)


# In[81]:


feature_sorted = sorted(featImport, key=lambda x: x[1], reverse=True)


# In[82]:


print feature_sorted


# In[129]:


# function that takes a input dataset without the churn column and returns predictions, probabilities
@numba.jit
def predictCustChurn(x):
    x = pd.read_json(x,orient='split')
    print x
    pred = rcClassifier.predict(x)
    probs = rcClassifier.predict_proba(x)
    return {'pred':pred,'probs':probs}
    
    


# In[130]:


retVal = predictCustChurn(churnData_test[features][:10].to_json(orient='split'))


# In[131]:


print retVal['pred']


# In[98]:




