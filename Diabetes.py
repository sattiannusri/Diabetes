#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score


# ## Reading the data

# #Data can be downloaded from:
# #https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset

# In[67]:


diab=pd.read_csv('D:/R programs/diabetes_data_upload.csv')
diab.head()


# ## Exploratory Data Analysis

# In[68]:


diab['Age'].describe()


# In[69]:


diab.isnull().sum()


# In[70]:


diab['class'].value_counts()


# In[71]:


sns.set_style('whitegrid')
sns.countplot(x='class',data=diab,palette='rainbow')


# In[72]:


sns.set_style('whitegrid')
sns.countplot(x='class',hue='Polyuria',data=diab,palette='rainbow')


# In[73]:


sns.set_style('whitegrid')
sns.countplot(x='class',hue='Polydipsia',data=diab,palette='rainbow')


# In[74]:


diab.corr()


# In[75]:


def age(df):
    if df<=25:
        return 'Young'
    elif df>25 and df<60:
        return'Middle'
    else:
        return 'Old'


# In[76]:


diab['Age']=diab['Age'].apply(age)


# In[77]:


diab.head()


# ## Creating dummy variables for categorical features

# In[78]:


diab.columns


# In[79]:


diab=pd.get_dummies(diab,columns=['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity'])


# In[80]:


diab.head()


# # Label encoding the target variable

# In[81]:


label=preprocessing.LabelEncoder()
diab['class']=label.fit_transform(diab['class'])
diab.head()


# ## Building models

# In[82]:


x,y=diab.drop(['class'],axis=1),diab['class']


# ## Train-Test Split

# In[83]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# ## Defining models

# In[84]:


log=LogisticRegression()
rf=RandomForestClassifier()
ada=AdaBoostClassifier()
bag=BaggingClassifier()
xtra=ExtraTreesClassifier()
gb=GradientBoostingClassifier()


# ## Fitting the models

# In[85]:


log.fit(x_train,y_train)
rf.fit(x_train,y_train)
ada.fit(x_train,y_train)
bag.fit(x_train,y_train)
xtra.fit(x_train,y_train)
gb.fit(x_train,y_train)


# ## Making predictions

# In[86]:


log_pred=log.predict(x_test)
rf_pred=rf.predict(x_test)
ada_pred=ada.predict(x_test)
bag_pred=bag.predict(x_test)
xtra_pred=xtra.predict(x_test)
gb_pred=gb.predict(x_test)


# ## Evaluation of models

# In[87]:


print('Accuracy of Logistic regression model is {:.2f}'.format(accuracy_score(y_test,log_pred)))
print('Accuracy of Random forest model is {:.2f}'.format(accuracy_score(y_test,rf_pred)))
print('Accuracy of Adaboost model is {:.2f}'.format(accuracy_score(y_test,ada_pred)))
print('Accuracy of Bagging model is {:.2f}'.format(accuracy_score(y_test,bag_pred)))
print('Accuracy of Extra trees model is {:.2f}'.format(accuracy_score(y_test,xtra_pred)))
print('Accuracy of Gradient boosting model is {:.2f}'.format(accuracy_score(y_test,gb_pred)))


# In[88]:


print('Precision of Logistic regression model is {:.2f}'.format(precision_score(y_test,log_pred)))
print('Precision of Random forest model is {:.2f}'.format(precision_score(y_test,rf_pred)))
print('Precision of Adaboost model is {:.2f}'.format(precision_score(y_test,ada_pred)))
print('Precision of Bagging model is {:.2f}'.format(precision_score(y_test,bag_pred)))
print('Precision of Extra trees model is {:.2f}'.format(precision_score(y_test,xtra_pred)))
print('Precision of Gradient boosting model is {:.2f}'.format(precision_score(y_test,gb_pred)))


# ## Extracting important features

# In[89]:


imp=pd.DataFrame({'Feature':list(x_train.columns),'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
imp


# ## Conclusion: Polyuria and Polydipsia are the significant features

# In[ ]:




