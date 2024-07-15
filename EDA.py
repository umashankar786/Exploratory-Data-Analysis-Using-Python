#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv("data_clean.csv")


# In[4]:


df


# In[5]:


df.info()


# In[6]:


#drop Temp c column

df.drop("Temp C",axis=1,inplace=True)


# In[7]:


df


# In[8]:


df["Month"] = pd.to_numeric(df["Month"],errors = "coerce") # convert month column to numeric

#incase obj/strings are found replcae them with "NAN" missing values


# In[9]:


#rename

df = df.rename(columns = {"Solar.R":"Solar","Unnamed: 0":"index"})


# In[10]:


df


# In[11]:


#missing values

df.isna().sum()


# In[12]:


#miss/total *100

(df.isna().sum())/(len(df))*100


# In[13]:


#handle missing values
#fillna()

#ozone



# In[14]:


sns.distplot(df["Ozone"])


# In[15]:


df["Ozone"].fillna(df["Ozone"].median(),inplace=True)


# In[16]:


df["Ozone"].isna().sum()


# In[17]:


sns.distplot(df["Solar"])


# In[18]:


df["Solar"].fillna(df["Solar"].mean(),inplace=True)


# In[19]:


df["Month"].unique()


# In[20]:


df["Month"].fillna(df["Month"].mode()[0],inplace=True)


# In[21]:


df["Weather"].fillna(df["Weather"].mode()[0],inplace=True)


# In[22]:


df.isna().sum()


# In[23]:


df["Weather"].unique()


# In[24]:


df1 = df.copy()


# In[25]:


ohe = pd.get_dummies(df["Weather"])


# In[26]:


ohe


# In[27]:


df = pd.concat([df,ohe],axis=1)


# In[28]:


df


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[30]:


df1["Weather"] = le.fit_transform(df1["Weather"])


# In[31]:


df1


# In[32]:


#Feature scaling


# In[33]:


#StandardScaler: Standardization
from sklearn.preprocessing import StandardScaler
se = StandardScaler()


# In[34]:


df[["Ozone","Solar","Wind","Temp"]] = se.fit_transform(df[["Ozone","Solar","Wind","Temp"]])


# In[35]:


df


# In[36]:


#MinMaxScaler: Normalization
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()


# In[37]:


df1[["Ozone","Solar","Wind","Temp"]] = ms.fit_transform(df1[["Ozone","Solar","Wind","Temp"]])


# In[38]:


df1


# In[39]:


#Outliers handling

sns.boxplot(df["Ozone"])


# In[40]:


sns.boxplot(df["Solar"])


# In[41]:


sns.boxplot(df["Wind"])


# In[42]:


sns.boxplot(df["Temp"])


# In[43]:


#Ozone and Wind have outliers


# In[44]:


# IQR handling for Ozone

q1 = np.percentile(df["Ozone"],25)
q3 = np.percentile(df["Ozone"],75)

iqr = q3-q1

ub = q3+1.5*iqr
lb = q1-1.5*iqr


# In[45]:


df[df["Ozone"]>ub]


# In[46]:


df["Ozone"][df["Ozone"]>ub] = ub
df["Ozone"][df["Ozone"]<lb] = lb


# In[47]:


sns.distplot(df["Ozone"])


# In[48]:


sns.boxplot(df["Ozone"])


# In[49]:


# IQR handling for Wind

q1 = np.percentile(df["Wind"],25)
q3 = np.percentile(df["Wind"],75)

iqr = q3-q1

ub = q3+1.5*iqr
lb = q1-1.5*iqr


# In[50]:


df[df["Wind"]>ub]


# In[51]:


df["Wind"][df["Wind"]>ub] = ub
df["Wind"][df["Wind"]<lb] = lb


# In[52]:


sns.boxplot(df["Wind"])


# In[53]:


#removing the outliers

q1 = np.percentile(df1["Ozone"],25)
q3 = np.percentile(df1["Ozone"],75)

iqr = q3-q1

ub = q3+1.5*iqr
lb = q1-1.5*iqr


# In[54]:


new_df = df1[df1["Ozone"]<ub]


# In[55]:


len(new_df)


# In[56]:


sns.distplot(new_df["Ozone"])


# In[57]:


sns.boxplot(new_df["Ozone"])


# In[58]:


q1 = np.percentile(df1["Wind"],25)
q3 = np.percentile(df1["Wind"],75)

iqr = q3-q1

ub = q3+1.5*iqr
lb = q1-1.5*iqr


# In[59]:


new_df = new_df[new_df["Wind"]<ub]


# In[60]:


sns.boxplot(new_df["Wind"])


# In[61]:


df.info()


# In[62]:


df.columns


# In[63]:


df["Month"].value_counts()


# In[64]:


sns.countplot(x="Month",data=df)


# In[65]:


df["Year"].value_counts()


# In[66]:


df["Month"].value_counts()


# In[67]:


df2 = df["Month"].value_counts().rename_axis('Month').reset_index(name="Counts")


# In[68]:


df2


# In[69]:


sns.barplot(y =df2["Counts"],x= df2["Month"])


# In[70]:


plt.pie(df2["Counts"],labels=df2["Month"])


# In[71]:


#pip install pandas-profiling


# In[72]:


#import pandas_profiling as pp


# In[79]:


df_new = pd.read_csv("data_clean.csv")


# In[74]:


#eda_report = pp.ProfileReport(df_new)


# In[75]:


#eda_report.to_file(output_file = "eda_report1.html")


# In[76]:


pip install sweetviz


# In[77]:


import sweetviz as sv


# In[80]:


sv_report = sv.analyze(df_new)


# In[81]:


sv_report.show_html("eda_report2.html")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




