
# coding: utf-8

# <h1>Predicting Customer Churn in a Telecom Company</h1>
# <p>This notebook is to demonstrate a simple machine learning problem using Jupyter notebooks and the datascience.com platform. The dataset that I've used in this example is from telecom customer data set which can be downloaded <a href="http://www.dataminingconsultant.com/data/churn.txt"> here </a> </p>
# <p>Each record in this dataset is a customer of this company and has customer attributes such as phone number, call minutes and so on. The dependent variable is 'Churn', which indicates if the customer is still a customer or has cancelled the service. And as expected - our goal in this exercise is to predict if the customer will churn or not, based on the attributes that are readily available to us.</p>

# # Importing the data

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve
import numba
from matplotlib import pyplot as plt
import json


# In[2]:


# get_ipython().magic(u'matplotlib inline')


# ### Let's take a look at what columns are part of the dataset and what a sample of the dataset looks like :

# Use pandas read_csv function to read the file from the url above. 
# ``` python
# churnData = pd.read_csv('http://www.dataminingconsultant.com/data/churn.txt')
# ```

# In[3]:


churnData = pd.read_csv('http://www.dataminingconsultant.com/data/churn.txt')


# Columns in the dataset : 

# In[4]:


for i in churnData.columns.tolist():
    print(i)


# A sample of the dataset : 

# In[5]:


churnData.head(10)


# Check the column datatypes : 

# In[6]:


print(churnData.dtypes)


# In[7]:


# convert the ones with object to string 
obj_col = churnData.select_dtypes(include =[np.object]).columns.tolist()


# In[39]:





# In[8]:


churnData[obj_col] = churnData[obj_col].astype('str')


# In[9]:


print(churnData.dtypes)


# ## Exploratory analysis
# Let's do some exploratory analysis to learn more about the dataset.
# We'll start with looking at some preliminary distributions. 
# 

# In[10]:


# sns.factorplot( 'Churn?', data=churnData, kind = 'count')


# The above visualization shows the distribution of the churned customers vs the non-churned customers in the dataset. Now lets compare the other variables in the dataset against churn? and see if it is indicative of anything. 

# In[66]:


# sns.set_style("whitegrid")
# sns.boxplot(x="Churn?", y="Account Length", data=churnData)


# It's interesting to observe that the account duration has no effect on the customers decision to churn. 

# In[11]:


# sns.set_style("whitegrid")
# sns.boxplot(x="Churn?",y="CustServ Calls",data=churnData)


# We see that the range of average number of customer service calls by customers who ended up churning are a bit wider than the ones that didn't end up churning. Though the results isn't statistically significant at this point, this could be a feature that will help us predict if a customer churns or not

# We could continue our exploration and draw plots with other variables to observe their relationship amongst the independent and dependent variables. But for the sake of this demo, i'll stop here and proceed to predictive modelling stage. 

# ## Preparing the data for modelling 
# In this example, we'll use the random forest model to predict if the customer churns or not. We need to prepare our dataset for modelling, for example, change strings into numeric factors that the model can understand or remove variable that we don't think will be useful in the modelling process. 
# 
# The dependent variable is presently 'True.' or 'False.' We need to change this to 1 and 0 to give a boolean representation to this. 

# In[12]:


churnData['Churn'] = churnData['Churn?'].map(lambda x: 1 if x=='True.' else 0)


# Let's remove the variables that we think won't be useful in this example. In this case, we'll remove the Area Code ,Phone, State. We'll also remove the Churn? variable now that we have encoded the value in the 'Churn' variable instead. 

# In[13]:


# remove columns that we won't use 
churnData1 = churnData.drop(['Area Code','Phone','State','Churn?'],axis=1)


# Convert the object variables and create dummy variables. 

# In[40]:


# find out which columns are object type and convert them to dummy vars
list_ = list(churnData1.select_dtypes(include=[np.object]).columns)
churnData2 = pd.get_dummies(churnData1, prefix=list_)


# Let's take a look at the data set now and them combine them with the rest of the dataset

# In[41]:


churnData2.head()


# In[16]:


# # merge with the other columns
# list_nonObj = list(churnData.select_dtypes(exclude=['object']).columns)
# churnData3 = churnData[list_nonObj]


# In[44]:


inputData = churnData2


# ### Final Input Dataset
# This is what our final input dataset looks like. 

# In[45]:


inputData.head()


# We'll split the dataset into training and testing datasets. 

# In[46]:


# split the input data set into train and test 
churnData_train, churnData_test = train_test_split(inputData, test_size =0.3)


# Get the list of input features

# In[47]:


features = inputData.drop(['Churn'], axis=1).columns


# ### Training the model 
# We'll pass the input dataset to the random forest classifier to generate the model that will predict customer churn.

# In[48]:


rcClassifier = RandomForestClassifier(n_estimators=40)
rcClassifier.fit(churnData_train[features], churnData_train['Churn'])


# ### Making Predictions
# We'll pass our test set into the model to get the predictions and the probabilty associated with each prediction

# In[49]:


# Make Predictions
predictions = rcClassifier.predict(churnData_test[features])
probabilities = rcClassifier.predict_proba(churnData_test[features])


# In[50]:


score = rcClassifier.score(churnData_test[features],churnData_test['Churn'])
print "Accuracy : ", score


# In[53]:


featureImportances = rcClassifier.feature_importances_
featureImportanceIndex = np.argsort(rcClassifier.feature_importances_)[::-1]
featureLabels = churnData_test.columns


# In[55]:


columnsImportant = churnData_test.iloc[:,featureImportanceIndex.tolist()].columns


# ## Plotting the feature importances 
# Let's take a look at what feature is important to predict if a customer will churn. 

# In[57]:


# plt.title("Feature Importances")
# plt.barh(range(10), featureImportances[featureImportanceIndex][:10][::-1], color='b', align='center')
# plt.yticks(range(10), columnsImportant[:10][::-1]) 
# plt.xlabel('Relative Importance')
# plt.show()


# We observe that day charge,  day minutes, customer service calls  are the some of the important features that the model suggests. Let's plot the day charge, evening charge, and night charge and observe if there is any difference between them.

# In[65]:


# sns.boxplot(y='Day Charge', x = 'Churn?', data=churnData)
# sns.boxplot(y='Eve Charge', x = 'Churn?', data=churnData,color='g')
# sns.boxplot(y='Night Charge', x = 'Churn?', data=churnData, color ='r')


# ### Recommendations 
# 1. Reduce day charges. 
# 2. Bring on some loyalty program

# In[ ]:


# function that takes a input dataset without the churn column and returns predictions, probabilities
@numba.jit
def predictCustChurn(x):
    x = pd.read_json(x,orient='split')
    print x
    pred = rcClassifier.predict(x)
    probs = rcClassifier.predict_proba(x)
    result = pd.Series(pred).to_json(orient='split')
    #     return json.dumps(result)
    return "heres a response"