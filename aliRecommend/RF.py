
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:

subset = pd.read_csv("tianchi_fresh_comp_train_item.csv")


# In[3]:

train = pd.read_csv("train_dataset/train1.csv")
valid = pd.read_csv("predict_dataset/predict_2.csv")
test = pd.read_csv("predict_dataset/predict_3.csv")
# valid = pd.read_csv("train_dataset/train2.csv")
# test = pd.read_csv("train_dataset/train3.csv")


# In[4]:

valid_num = len(pd.read_csv("label_dataset/labelDataset_2.csv"))
test_num = len(pd.read_csv("label_dataset/labelDataset_3.csv"))


# In[5]:

# import pickle
# with open("model/xgb/feature_top46.pkl", "rb") as f:
#     important_feature = pickle.load(f)


# In[6]:

# important_feature = ['U_b28day_buy',
#  'U_b4day_browse',
#  'b7day_UI_U_browse_rate',
#  'b28day_UI_U_browse_rate',
#  'U_b7day_browse',
#  'U_b28day_browse',
#  'U_b28day_cart',
#  'U_b1day_browse',
#  'U_b2day_browse',
#  'b4day_UI_U_browse_rate',
#  'C_b28day_buy',
#  'b28day_UC_U_browse_rate',
#  'b2day_UI_U_browse_rate',
#  'b28day_UC_C_browse_rate',
#  'U_b7day_cart',
#  'b7day_UC_U_browse_rate',
#  'b1day_I_C_browse_rate',
#  'b28day_I_C_browse_rate',
#  'b4day_UC_U_browse_rate',
#  'b28day_UI_UC_browse_rate',
#  'UI_b28day_browse',
#  'b1day_UI_U_browse_rate',
#  'b1day_UC_C_browse_rate',
#  'U_b28day_collect',
#  'b28day_UI_U_cart_rate',
#  'b1day_UC_U_browse_rate',
#  'b7day_UC_C_browse_rate',
#  'b28day_UI_I_browse_rate',
#  'UC_b28day_browse',
#  'U_b7day_buy',
#  'b28day_UI_U_collect_rate',
#  'I_b28day_browse',
#  'C_b1day_buy',
#  'b7day_I_C_browse_rate',
#  'b4day_I_C_browse_rate',
#  'b2day_UC_U_browse_rate',
#  'U_b4day_collect',
#  'b2day_UC_C_browse_rate',
#  'b2day_I_C_browse_rate',
#  'U_b4day_cart',
#  'b4day_UC_C_browse_rate',
#  'I_b28day_buy',
#  'C_b2day_buy',
#  'b28day_I_C_cart_rate',
#  'b28day_UC_U_cart_rate',
#  'U_b4day_buy']


# In[7]:

X_train = train.loc[:,"U_b1day_browse":].values
# X_train = train[important_feature].values
y_train = train.label.values


# In[8]:

X_valid = valid.loc[:,"U_b1day_browse":].values
# X_valid = valid[important_feature].values
y_valid = valid.label.values


# In[9]:

X_test = test.loc[:,"U_b1day_browse":].values
y_test = test.label.values


# In[10]:

min_max_scaler = MinMaxScaler()


# In[11]:

X_train = min_max_scaler.fit_transform(X_train)
X_valid = min_max_scaler.transform(X_valid)
X_test = min_max_scaler.transform(X_test)


# In[12]:

valid = valid[["user_id", "item_id", "label"]]
test = test[["user_id", "item_id", "label"]]


# In[13]:

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[29]:

def modelfit(alg, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
                
#     #Print Feature Importance:
#     if printFeatureImportance:
#         feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
#         feat_imp.plot(kind='bar', title='Feature Importances')
#         plt.ylabel('Feature Importance Score')


# In[35]:

rf0 = RandomForestClassifier(criterion='gini')
modelfit(rf0)


# In[43]:

param_test1 = {'n_estimators':range(150,1000,100)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=2,
                                  min_samples_leaf=1,max_depth=6,max_features='sqrt',max_leaf_nodes= 2000,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)


# In[44]:

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[48]:

param_test2 = {'min_samples_leaf':[1,2]}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=2,n_estimators=150,
                                  max_depth=6,max_features='sqrt',max_leaf_nodes= 2000,random_state=10), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train, y_train)


# In[49]:

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[52]:

param_test3 = {'max_leaf_nodes':range(300, 200)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=2,n_estimators=150,
                                  min_samples_leaf=1,max_depth=6,max_features='sqrt',random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train, y_train)


# In[53]:

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[58]:

param_test4 = {'max_features':[14,15,16]}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=2,n_estimators=150,max_leaf_nodes=300,
                                  min_samples_leaf=1,max_depth=6,random_state=10), 
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train, y_train)


# In[59]:

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[64]:

param_test5 = {'min_samples_split':[1,2]}
gsearch5 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=150,max_leaf_nodes=300,max_features=15,
                                  min_samples_leaf=1,max_depth=6,random_state=10), 
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train, y_train)


# In[65]:

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[68]:

param_test6 = {'max_depth':[9,10,12,14]}
gsearch6 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=150,max_leaf_nodes=300,max_features=15,min_samples_split=1,
                                  min_samples_leaf=1,random_state=10), 
                       param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train, y_train)


# In[69]:

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[14]:

model = RandomForestClassifier(n_estimators=150,max_leaf_nodes=300,max_features=15,min_samples_split=1,
                                  max_depth=10,min_samples_leaf=1,random_state=10)


# In[15]:

# Model Report
# Accuracy : 0.9932
# AUC Score (Train): 0.999724
# CV Score : Mean - 0.8837301 | Std - 0.004197822 | Min - 0.878843 | Max - 0.8890626


# In[16]:

model.fit(X_train, y_train)


# In[17]:

train_feat = model.predict_proba(X_train)[:,1]
valid_feat = model.predict_proba(X_valid)[:,1]
test_feat = model.predict_proba(X_test)[:,1]
train_feat = pd.DataFrame({"prob_rf":train_feat})
valid_feat = pd.DataFrame({"prob_rf":valid_feat})
test_feat = pd.DataFrame({"prob_rf":test_feat})
train_feat.to_csv("model/rf/prob_feat/train_feat.csv", index=False)
valid_feat.to_csv("model/rf/prob_feat/valid_feat.csv", index=False)
test_feat.to_csv("model/rf/prob_feat/test_feat.csv", index=False)


# In[74]:

preds = model.predict_proba(X_valid)[:,1]
index = preds.argsort()[-500:]
preds = valid.loc[index]
preds = preds[preds.item_id.isin(subset.item_id)]
p = len(preds[preds.label == 1]) / float(len(preds))
r = len(preds[preds.label == 1]) / float(valid_num)
print(len(preds[preds.label == 1]))
print(len(preds))
print((2 * p * r) / (p + r))


preds = model.predict_proba(X_test)[:,1]
index = preds.argsort()[-500:]
preds = test.loc[index]
preds = preds[preds.item_id.isin(subset.item_id)]
p = len(preds[preds.label == 1]) / float(len(preds))
r = len(preds[preds.label == 1]) / float(test_num)
print(len(preds[preds.label == 1]))
print(len(preds))
print((2 * p * r) / (p + r))


# In[18]:

offline_train = pd.read_csv("train_dataset/train3.csv")
online_predict = pd.read_csv("predict_dataset/predict_4.csv")


# In[19]:

offline_X_train = offline_train.loc[:,"U_b1day_browse":].values
offline_y_train = offline_train.label.values
online_test = online_predict.loc[:,"U_b1day_browse":].values


# In[20]:

offline_X_train = min_max_scaler.transform(offline_X_train)
online_test = min_max_scaler.transform(online_test)


# In[21]:

model.fit(offline_X_train, offline_y_train)


# In[22]:

offline_train_prob = model.predict_proba(offline_X_train)[:,1]
offline_train_prob = pd.DataFrame({"prob_rf":offline_train_prob})
offline_train_prob.to_csv("model/rf/prob_feat/offline_train_feat.csv", index=False)


# In[79]:

preds = model.predict_proba(online_test)[:,1]


# In[80]:

preds = pd.DataFrame({"gbdt":preds})


# In[83]:

preds.to_csv("model/rf/predict_prob.csv", index=False)


# In[68]:

index = preds.argsort()[-500:]


# In[69]:

submit_result = online_predict.loc[index]


# In[70]:

submit_result = submit_result[["user_id", "item_id"]]


# In[71]:

submit_result.to_csv("model/gbdt/tianchi_mobile_recommendation_predict.csv", index=False)


# In[ ]:



