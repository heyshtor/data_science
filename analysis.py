#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ip2geotools
from ip2geotools.databases.noncommercial import DbIpCity

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import plot_confusion_matrix,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[36]:


# Load the data from the .json file into a dataframe
df = pd.read_json ("customers.json", lines = True)
pd.set_option('max_colwidth', -1)

df.head(15)


# In[37]:


# Count how many fraudulent and non-fraudulent transactions there are (including the items with missing values)
df['fraudulent'].value_counts(dropna=False)
# The data is relatively balanced


# In[38]:


# Let's look at the fraudulent transactions
df[df.fraudulent == True]
# They seem to have:
# multiple orders, sometimes to the address that doesn't match the billing address;
# different payment methods,
# failed transactions,
# sometimes there's data on payment methods but not on transactions.


# In[39]:


# Based on this, let's design some features:

# Main function for the features 
def extract_features(df):
    df_features = pd.DataFrame(columns=['Fraud','email','realIP','Billing_address_matches_IP','Billing_address_in_delivery','Average transaction','Max transaction','Min transaction','Number of transactions','Percentage of failed transactions','Payment methods count', 'Number of unique payment methods'])
    for i in range(len(df)):
        # data on the subject is line of the dataframe
        subj = df.iloc[i,:]
        # feature set 1 - check if the IP address is a real one and whether it matches the billing address, and if at least one delivery address matches the billing address:
        [realIP,Billing_address_matches_IP] = IP_match_delivery_address(subj)
        billing_address_in_delivery = billing_match_delivery(subj)
        # feature set 2 - average amount spent, minimum and maximum transactions, total number of transactions, percentage of failed transactions:
        [av,max_,min_,number,failed_transactions] = transaction_features(subj)
        # feature set 3 - total number of payment methods used, and number of unique payments:
        [payment_methods_count, unique_payment] = payment_methods_features(subj)

        # append the features to the features' dataframe
        to_append = [ subj.fraudulent, subj.customer['customerEmail'],realIP,Billing_address_matches_IP,billing_address_in_delivery,av,max_,min_,number,failed_transactions,payment_methods_count,unique_payment]
        df_length = len(df_features)
        df_features.loc[df_length] = to_append
        print('Appended subject number',i,' out of',len(df))    
    return df_features

# Function to determine whether the user IP matches the billing address
def IP_match_delivery_address(subj):
    IP = subj.customer['customerIPAddress']
    delivery = subj.customer['customerBillingAddress']
    
    # extract location data from the IP
    try:
        response = DbIpCity.get(IP, api_key='free')
    except:
        return 0,0
    # get the city, country and region
    city = str(response.city).split()[0]
    country = response.country
    reg = response.region
    # check if either city/region/country is in the delivery adddress:
    if (city in delivery) or (country in delivery) or (reg in delivery):
        return 1,1
    else:
        return 1,0

# Function to determine whether the user billing address matches their delivery address on at least one of the orders
def billing_match_delivery(subj):
    orders = subj.orders
    billing = subj.customer['customerBillingAddress']
    delivery = []
    
    # if the orders exist, iterate over them and form a list of delivery addresses
    if len(orders) != 0:  
        for i in subj.orders:
            for key, value in i.items():
                delivery.append(i['orderShippingAddress'])
        # if billing address is in the list of addresses, return 1, otherwise 0
        if billing in delivery:
            return 1
        return 0
    # if there's no information on the orders, return NAN
    return np.NAN

# Function for the transaction features:
# average amount spent, minimum and maximum transactions, total number of transactions, percentage of failed transactions
def transaction_features(subj):
    tr = subj.transactions
    
    # if transactions exist, iterate over them to determine the minimum and maximum, and count the failed transactions
    if len(tr)!=0 :
        av=0
        max_ = -1
        min_ = 50000
        number = 0
        failed_ = 0
        
        for i in range(len(tr)):
            av = av+ tr[i]['transactionAmount']
            if tr[i]['transactionAmount'] > max_ : max_ = tr[i]['transactionAmount']
            if tr[i]['transactionAmount'] < min_ : min_ = tr[i]['transactionAmount']
            if tr[i]['transactionFailed'] == True : failed_ += 1
        # count average, total, and the percentage of failed transactions
        av = av/(len(tr))
        number = len(tr)
        failed_transactions = float(failed_ / number)

        return av,max_,min_,number,failed_transactions
    else:
        # if there's no data on transactions, return NANs
        return np.NAN,np.NAN,np.NAN,np.NAN,np.NAN

# Function for the payment features:
# total number of payment methods used, and number of unique payments
def payment_methods_features(subj):
    # total # of payments
    payment_methods_count = str(subj.paymentMethods).count("paymentMethodId")
    
    # # of unique payment methods
    payments_list = []  
    for i in subj.paymentMethods:
        for key, value in i.items():
            if not i['paymentMethodType'] in payments_list:
                payments_list.append(i['paymentMethodType'])
    unique_payment = len(payments_list)
    return payment_methods_count, unique_payment
        
df_features = extract_features(df)


# In[40]:


df_features.head(10)


# In[41]:


# Normalize features to 0-1 range

def normalize_feat(df_features):
    x = df_features.iloc[:,2:-1].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    features_norm = pd.DataFrame(x_scaled)
    return features_norm

features_norm = normalize_feat(df_features)

# Use imputer to substitute the NAN values for the mean value of the column:
features_norm = features_norm.fillna(np.nan)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(features_norm)

# Transform complete normalised features
features_norm = np.array( imp.transform(features_norm) )


# In[42]:


# Split the data into test (30%) and train:
X_train, X_test, y_train, y_test = train_test_split(features_norm, df_features.iloc[:,0], test_size=0.3)

# Train the model. I chose SVM because is seems to be a common practice for transaction fraud:
clf = svm.SVC(kernel = 'poly')
clf.fit(X_train, list(y_train))


# In[43]:


# Test the model:
pred = clf.predict(X_test) 


# In[44]:


# Plot the confusion matrix
disp = plot_confusion_matrix(clf, X_test, list(y_test) )
conf_mat = confusion_matrix(pred, list(y_test))
acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
plt.title('Accuracy:' + str(acc))
plt.show()

# If we run the code several times, the accuracy that the model returns is in the 70-80% range


# In[45]:


# Now let's look at other classifiers to select the best model for the task. 
# I went for the RandomForestClassifier (also a suitable model for this type of data), MultinomialNB, and LogisticRegression.

models = [
    RandomForestClassifier(n_estimators=600, max_depth=3, random_state=0),
    svm.SVC(kernel = 'poly'),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# cross-validation: test how good our models are on 1/5 of the data
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
# use normalised features from the dataframe as features and True/False fraud labels as labels
features = features_norm
labels = list(df_features.iloc[:,0])

# count accuracies for every model in the list
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  print ('Model compiled ok')

# create a datafrabe with cross-falidation results
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# plot the chart for the models
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[46]:


# Accuracy numbers
cv_df.groupby('model_name').accuracy.mean()
# Our chosen model performs well, and the data isn't spread out too broadly, it's relatively stable.


# In[47]:


# Now let's select the most important features of the model

classifier = clf

# select 3 best features by using backward feature selection.
sfs1 = SFS(classifier,k_features=3, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(f, l)
sfs1.subsets_

# If we fun this code several times and experiment with the # of features (k_features parameter)
# and forward and backward feature selection (forward parameter), we'll see that the features that appear frequently 
# are 3 (amount of average transaction), 4 (max transaction), and 8 (# of unique payment methods).


# In[48]:


# Plot the chart for the model performance and the number of features it has
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
# Out of the features created, 3 features are optimal for determining whether the user is traudulent or not.

# Potential areas of exploration and designing features are:
# looking at the payment method provider and payment method issuer,
# failed transactions with respect to the payment method;
# state of the orders.


# In[ ]:




