#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
train_df = pd.read_csv("train.csv") 
#print(train_df) #there are 56 attributes in data and 15120 rows in data


# In[2]:


print(train_df.dtypes) # all are numeric categorical data
print(train_df.shape)


# In[3]:


train_df.isnull().sum()
#There are no null values in data


# In[98]:


pip install seaborn


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(train_df.Cover_Type)
plt.show()
#each cover type has same amount of records


# In[6]:


sns.set_style('whitegrid') 
sns.distplot(train_df.Elevation,bins=30)
plt.show()
#elevation has more number of records in range of 2000-3000


# In[4]:


test_df = pd.read_csv("test.csv") 


# In[11]:


sns.set_style('whitegrid') 
sns.distplot(train_df["Aspect"])
plt.show()
#aspect has more number of records in range of (0-100)


# In[17]:


sns.set_style('whitegrid') 
sns.distplot(train_df["Hillshade_3pm"])
plt.show()
#more data has hillsahde index at 3pm around 150  


# In[7]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Hillshade_9am"])
plt.show()
#hillshade is more 


# In[8]:


sns.set_style('whitegrid') 
sns.distplot(train_df["Hillshade_Noon"])
plt.show()


# In[9]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Horizontal_Distance_To_Fire_Points"])
plt.show()


# In[10]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Horizontal_Distance_To_Hydrology"])
plt.show()


# In[11]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Horizontal_Distance_To_Roadways"])
plt.show()


# In[12]:


train_attributes=list(train_df.columns.values)


# In[13]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Vertical_Distance_To_Hydrology"])
plt.show()


# In[14]:


sns.set_style('whitegrid')  
sns.distplot(train_df["Slope"])
plt.show()


# In[16]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Elevation",data=train_df)
plt.show()
#cover type 4 has lowest Elevation whereas cover type 1 and 7 has highest elevation


# In[17]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Aspect",data=train_df)
plt.show()
#allcover types have almost same aspect ratio 


# In[18]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Slope",data=train_df)
plt.show
#cover type 3 and 4 have highest slope


# In[19]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Hillshade_3pm",data=train_df)
plt.show()
#hillshade at 3pm is almost equal fro all covertypes 


# In[20]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Hillshade_9am",data=train_df)
plt.show()
#hillsahde at 9pm has equal records  for all covertypes


# In[21]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Hillshade_Noon",data=train_df)
plt.show()
#hillshade at 6pm has equal records for all covertypes


# In[22]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Vertical_Distance_To_Hydrology",data=train_df)
plt.show()


# In[23]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Hydrology",data=train_df)
plt.show()


# In[24]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Roadways",data=train_df)
plt.show()


# In[25]:


sns.set_style('whitegrid')  
sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points",data=train_df)
plt.show()


# In[7]:


attributes=["Elevation","Aspect","Slope","Hillshade_3pm","Hillshade_9am","Hillshade_Noon","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Horizontal_Distance_To_Fire_Points"]
numerical_df=train_df[attributes]
corr=numerical_df.corr()


# In[10]:


ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);#https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec


# In[2]:


train_df.iloc[:,[1,2,3,4,5,6,7,8,9,10]].skew()


# In[3]:


train_df.std(axis = 0) 


# In[3]:


#as standard deviation of col 15 and 17 is 0
train_df.drop(['Soil_Type15','Soil_Type7'],axis=1)


# In[8]:


#feature Importance
#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#https://www.geeksforgeeks.org/decision-tree-implementation-python/
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


X=train_df.values[:,1:55];
Y=train_df.values[:,55];

#print(X);
#print(Y);
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(X_train,y_train);
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(numerical_df, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
feature_importances


# In[10]:


import matplotlib.pyplot as plt
feature_list=[]
list1=[]
list2=[]
for i in feature_importances:
    k=0
    for j in i:
        if k==0:
            list1.append(j)
        if k==1:
            list2.append(j)
        k=k+1    
plt.style.use('ggplot')
plt.plot(list2,list1,label="feature importances")


# In[6]:


test_df=pd.read_csv("test.csv") 


# In[4]:


print(test_df.shape)


# In[7]:


from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt

height=[5,10,15,20,25]
acc_at_height_h=[]
test_new=test_df.values[:,1:55]
    
def Decision_Tree_Classifier(height):
    dec=DecisionTreeClassifier(max_depth=height)
    dec.fit(X_train,y_train)
    pred=dec.predict(X_test)
    acc=dec.score(X_test,y_test)
    acc_at_height_h.append(round(acc,3))
    #print(acc)
    dec.fit(X,Y)
    y_on_test=dec.predict(test_new)

for i in height:
    Decision_Tree_Classifier(i)

print(acc_at_height_h)
 
plt.style.use('ggplot')    
plt.plot(height,acc_at_height_h)
plt.xlabel="height"
plt.ylabel="Accuracy"
plt.show()    


# In[12]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

n_estimators=[50,100,200,250,300]
acc_of_estimator=[]
y_predicted=[]
def random_forest(estimator):
    rf=RandomForestClassifier(n_estimators=estimator,class_weight='balanced',n_jobs=2,random_state=42)
    rf.fit(X_train,y_train)
    y_predicted=[]
    pred=rf.predict(X_test)
    acc=rf.score(X_test,y_test)
    acc_of_estimator.append(round(acc,3))
    #print(acc)
    rf.fit(X,Y)
    y_on_test=rf.predict(test_new)
    y_predicted=y_on_test
    
for i in n_estimators:
    random_forest(i)
print(acc_of_estimator)
plt.style.use('ggplot')
plt.plot(n_estimators,acc_of_estimator)
plt.xlabel="Esimators"
plt.ylabel="Accuracy"
plt.show()    


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

neighbors=[1,2,3,4,5]
acc_at_neighbors=[]
test_new=test_df.values[:,1:55]
   

def knn(neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
    knn.fit(X_train, y_train)
    pred=knn.predict(X_test)
    acc=knn.score(X_test,y_test)
    acc_at_neighbors.append(round(acc,3))
    #print(acc)
    knn.fit(X,Y)
    #y_on_test=knn.predict(test_new)
    
for i in neighbors:
    knn(i) 
    
plt.style.use('ggplot')
print(acc_at_neighbors)
plt.plot(neighbors,acc_at_neighbors)
plt.xlabel="Neighbors"
plt.ylabel="Accuracy"
plt.show()     


# In[13]:


models = []
models.append(('KNN', KNeighborsClassifier(1)))
models.append(('DEC', DecisionTreeClassifier(max_depth=25)))
models.append(('RFC', RandomForestClassifier(n_estimators=100)))


# In[14]:


#https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
from sklearn import model_selection
results = []
names = []
seed=7
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f" % (name, cv_results.mean())
    print(msg)


# In[15]:


give_fig = plt.figure()
give_fig.suptitle('Algorithm Comparison')
ax = give_fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:




