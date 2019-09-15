#!/usr/bin/env python
# coding: utf-8

# In[38]:


#import packages
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

#imoprt data
data = pd.read_csv('Churn_Modelling.csv')

#check for missing values
data.head(10)
data.info()
data.isna()


# In[39]:


#convert categorical data
OneHotGeo = pd.get_dummies(data['Geography'])

for i in range(len(OneHotGeo.columns)):
    data.insert(i+4, list(OneHotGeo)[i], OneHotGeo.iloc[:, i])
    
OneHotGender = pd.get_dummies(data['Gender'])
for i in range(len(OneHotGender.columns)):
    data.insert(i+7, list(OneHotGender)[i], OneHotGender.iloc[:, i])

data = data.drop(['Geography', 'Gender'], axis=1)

#drop customer ID and surname
data = data.drop(['RowNumber','CustomerId', 'Surname'], axis=1)


# In[4]:


#check head and tail
data.head()
data.tail()


# In[5]:


data.describe()


# In[133]:


sns.pairplot(data, hue='Exited')


# In[14]:


#plots
sns.distplot(data.CreditScore)
sns.distplot(data.Age)
sns.distplot(data.Tenure)
sns.distplot(data.Balance)
sns.distplot(data.EstimatedSalary)


# In[66]:


#correlations
corr = data.corr()


# In[68]:


#correalation heatmap
sns.heatmap(corr)


# In[176]:


#balance/products and exited
#balance - number of products correlation -0.304180

#scatter plt
sns.scatterplot(x='Balance', y='NumOfProducts', hue='Exited', size='Balance', data=data, alpha=0.3)

#balance and exited
NoBalance = data.loc[data['Balance'] == 0]
NoBalance['Exited'].value_counts()
#3117, 500
HasBalance = data.loc[data['Balance'] != 0]
HasBalance['Exited'].value_counts()
#4846, 1537

NoBalance['Exited'].value_counts()[0]/NoBalance['Exited'].count()
HasBalance['Exited'].value_counts()[0]/HasBalance['Exited'].count()
##no balance has a higher rention rate(0.8617638927287807, 0.7592041359862134)

#products and exited 
#customer with 3-4 products
MultiProducts = data.loc[data['NumOfProducts']>2]
#customer with 1-2 products
LessProducts = data.loc[data['NumOfProducts']<3]

#correlations
MultiProducts.corr()
#numofproducts-exited 0.192502
#balance-exited 0.345945
LessProducts.corr()
#numofproducts-exited -0.260762
#balance-exited 0.115281

MultiProducts['Exited'].value_counts()
#46, 280
LessProducts['Exited'].value_counts()
#7917, 1757

#retention rate
MultiProducts['Exited'].value_counts()[0]/MultiProducts['Exited'].count()
LessProducts['Exited'].value_counts()[0]/LessProducts['Exited'].count()
##customers who purchased 3-4 products tend to exit compared to those who purchased 1-2
##retention rates(0.1411042944785276, 0.8183791606367583)
##customers who purchased 3-4 products have balance positively related to exit decisions (0.345945)


# In[293]:


#age/active and exited
#age - exited 0.285323

#plot
sns.scatterplot(x='Age', y='IsActiveMember', data=data, hue='Exited', size='Balance', alpha=0.2)
sns.distplot(IsActive.loc[IsActive['Exited'] == 1]['Age'], bins=40)
sns.distplot(NotActive.loc[NotActive['Exited'] == 1]['Age'], bins=40)

#by active
IsActive = data.loc[data['IsActiveMember'] == 1]
NotActive = data.loc[data['IsActiveMember'] == 0]

#correlations
IsActive.corr()
#age - exited 0.173498
NotActive.corr()
#age - exited 0.466271

IsActive['Exited'].value_counts()
#4416, 735
NotActive['Exited'].value_counts()
#3547, 1302
IsActive['Exited'].count()
#5151
NotActive['Exited'].count()
#4849
##active members have higher retention rate (0.8573092603377985, 0.7314910290781604)
##51.51% of the members are active

#before 45
IsActive.loc[IsActive['Age'] <= 45]['Exited'].value_counts()
#3527, 389
NotActive.loc[NotActive['Age'] <= 45]['Exited'].value_counts()
#3282, 691

#retention rate
IsActive.loc[IsActive['Age'] <= 45]['Exited'].value_counts()[0]/IsActive.loc[IsActive['Age'] <= 45]['Exited'].count()
NotActive.loc[NotActive['Age'] <= 45]['Exited'].value_counts()[0]/NotActive.loc[NotActive['Age'] <= 45]['Exited'].count()
##before 45 active members have higher rentention rate (0.9006639427987743, 0.8263277120563806)

#after 45
IsActive.loc[IsActive['Age'] > 45]['Exited'].value_counts()
#889, 346
NotActive.loc[NotActive['Age'] > 45]['Exited'].value_counts()
#265, 611

#retention rate
IsActive.loc[IsActive['Age'] > 45]['Exited'].value_counts()[0]/IsActive.loc[IsActive['Age'] > 45]['Exited'].count()
NotActive.loc[NotActive['Age'] > 45]['Exited'].value_counts()[0]/NotActive.loc[NotActive['Age'] > 45]['Exited'].count()
##after 45 not active members have very low retention rate (0.719838056680162, 0.3025114155251142)


# In[352]:


#country and balance/products/credit card
#fr, gr, sp - balance correalations -0.231329, 0.401110, -0.134892

#by country
France = data.loc[data['France'] == 1]
Germany = data.loc[data['Germany'] == 1]
Spain = data.loc[data['Spain'] == 1]

#scatter plots
sns.scatterplot(x='Balance', y='EstimatedSalary', data=France, size='Balance', hue='Exited', alpha=0.5)
sns.scatterplot(x='Balance', y='EstimatedSalary', data=Germany, size='Balance', hue='Exited',alpha=0.5)
sns.scatterplot(x='Balance', y='EstimatedSalary', data=Spain, size='Balance', hue='Exited', alpha=0.5)

#for each country:
#retention rate
France['Exited'].value_counts()
#4204, 810
Germany['Exited'].value_counts()
#1695, 814
Spain['Exited'].value_counts()
#2064, 413

#retention rate
France['Exited'].value_counts()[0]/France['Exited'].count()
Germany['Exited'].value_counts()[0]/Germany['Exited'].count()
Spain['Exited'].value_counts()[0]/Spain['Exited'].count()
##Germany has a lower retention rate(0.8384523334662943, 0.6755679553607015, 0.8332660476382721)
##salary structures are similiar and don't seem to affect balance

#no balance
FRNoBalance = France.loc[data['Balance'] == 0]
GRNoBalance = Germany.loc[data['Balance'] == 0]
SPNoBalance = Spain.loc[data['Balance'] == 0]

#no balance rate
FRNoBalance.France.count()/France.France.count()
#2418/5014
GRNoBalance.Germany.count()/Germany.Germany.count()
#0/2509
SPNoBalance.Spain.count()/Spain.Spain.count()
#1199/2477
##Germany has extremely low no balance rate(0.48224970083765456, 0.0, 0.4840532902704885)

#balance median
France['Balance'].median()
#62153.5
Germany['Balance'].median()
#119703.1
Spain['Balance'].median()
#61710.44
##Germany tends to have more balance

#products
France['NumOfProducts'].value_counts()
#(2514, 2367, 104, 29)/5014
France['NumOfProducts'].value_counts()/France['NumOfProducts'].count()
#(0.501396090945353, 0.47207818109293975, 0.020741922616673316, 0.0057838053450339055)
Germany['NumOfProducts'].value_counts()
#(1349, 1040, 96, 24)/2509
Germany['NumOfProducts'].value_counts()/Germany['NumOfProducts'].count()
#(0.5376644081307294, 0.41450777202072536, 0.038262255878836186, 0.009565563969709047)
Spain['NumOfProducts'].value_counts()
#(1221, 1183, 66, 7)/2477
Spain['NumOfProducts'].value_counts()/Spain['NumOfProducts'].count()
#(0.4929350020185709, 0.4775938635446104, 0.026645135244247074, 0.002825999192571659)
##similiar product purchasing structures

#credit card
France['HasCrCard'].value_counts()
#(1471, 3543)/5014
France['HasCrCard'].value_counts()/France['HasCrCard'].count()
#(0.29337854008775427, 0.7066214599122457)
Germany['HasCrCard'].value_counts()
#(718, 1791)/2509
Germany['HasCrCard'].value_counts()/Germany['HasCrCard'].count()
#(0.28616978876046234, 0.7138302112395377)
Spain['HasCrCard'].value_counts()
#(756, 1721)/2477
Spain['HasCrCard'].value_counts()/Spain['HasCrCard'].count()
#(0.3052079127977392, 0.6947920872022608)
##similiar credit card holding rates


# In[393]:


#gender
#female, male - exited correaltion 0.106512, -0.106512

#by gender
Female = data.loc[data['Female'] == 1]
Male = data.loc[data['Male'] == 1]

#plots
sns.scatterplot(x='Age', y='EstimatedSalary', data=Female, hue='Exited', size='Balance', alpha=0.3)
sns.scatterplot(x='Age', y='EstimatedSalary', data=Male, hue='Exited', size='Balance', alpha=0.3)

#retention rate
Female['Exited'].value_counts()
#3404, 1139
Male['Exited'].value_counts()
#4559, 898

Female['Exited'].value_counts()[0]/Female['Exited'].count()
Male['Exited'].value_counts()[0]/Male['Exited'].count()
##Male has higher retention rate (0.7492846136913933, 0.8354407183434122)

#age > 45 group
Female.loc[Female['Age'] > 45]['Age'].count()/Female['Age'].count()
#0.22782302443319394
Male.loc[Male['Age'] > 45]['Age'].count()/Male['Age'].count()
#0.19717793659519883
Male.loc[Male['Age'] > 45]['IsActiveMember'].value_counts()[1]/Male.loc[Male['Age'] > 45]['IsActiveMember'].count()
#0.6143122676579925
Female.loc[Female['Age'] > 45]['IsActiveMember'].value_counts()[1]/Female.loc[Female['Age'] > 45]['IsActiveMember'].count()
#0.5545893719806764
##Female has larger age > 45 portion and lower active rate for age > 45

#retention rate
Female.loc[Female['Age'] > 45]['Exited'].value_counts()
#500, 535
Male.loc[Male['Age'] > 45]['Exited'].value_counts()
#654, 422

Female.loc[Female['Age'] > 45]['Exited'].value_counts()[0]/Female.loc[Female['Age'] > 45]['Exited'].count()
Male.loc[Male['Age'] > 45]['Exited'].value_counts()[0]/Male.loc[Male['Age'] > 45]['Exited'].count()
##for age > 45, female has lower retention rate (0.4830917874396135, 0.6078066914498141)


# In[190]:


#tenure groups
LongTerm =data.loc[data['Tenure'] > 7]
MidTerm = data.loc[data['Tenure'] > 3][data['Tenure'] <= 7]
ShortTerm = data.loc[data['Tenure'] <= 3]

LongTerm['Exited'].value_counts()[0]/LongTerm['Exited'].count()
MidTerm['Exited'].value_counts()/MidTerm['Exited'].count()
ShortTerm['Exited'].value_counts()/ShortTerm['Exited'].count()
##similiar rentention rate for different tenure groups(0.7955182072829131, 0.8035535535535535, 0.7885877318116976)


# In[410]:


#credit score/credit card/salary
#plots
sns.scatterplot(x='Balance', y='HasCrCard', hue='Exited', size='CreditScore', data=data, alpha=0.3)
sns.scatterplot(x='CreditScore', y='Balance', hue='HasCrCard', size='NumOfProducts', data=data, alpha=0.3)
sns.scatterplot(x='Balance', y='EstimatedSalary', hue='Exited', size='CreditScore', data=data, alpha=0.3)
##no significant trend


# In[ ]:




