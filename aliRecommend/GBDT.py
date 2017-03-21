
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[17]:

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


# In[18]:

gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0)


# In[25]:

param_test1 = {'n_estimators':range(10,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.3, min_samples_split=500,
                                  min_samples_leaf=50,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)


# In[26]:

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[29]:

param_test2 = {'max_depth':range(8,11,1), 'min_samples_split':range(2700,3000,100)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.3, n_estimators=70,
                                                max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train, y_train)


# In[30]:

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[35]:

param_test3 = {'min_samples_split':range(2400,2600,50), 'min_samples_leaf':range(60,70,1)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.3, n_estimators=70,max_depth=9,
                                                    max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train, y_train)


# In[36]:

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[37]:

modelfit(gsearch3.best_estimator_)


# In[40]:

param_test4 = {'max_features':range(11,14,1)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.3, n_estimators=70,max_depth=9, 
                            min_samples_split=2500, min_samples_leaf=63, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train, y_train)


# In[41]:

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[42]:

#Grid seach on subsample and max_features
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.3, n_estimators=70,max_depth=9, 
                            min_samples_split=2500, min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12),
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train, y_train)


# In[43]:

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[44]:

gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.3, n_estimators=70,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_1)


# In[45]:

gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=210,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_2)


# In[46]:

gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=420,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_3)


# In[47]:

gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.03, n_estimators=630,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_4)


# In[48]:

gbm_tuned_5 = GradientBoostingClassifier(learning_rate=0.02, n_estimators=1050,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_5)


# In[49]:

gbm_tuned_6 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=2100,max_depth=9, min_samples_split=2500, 
                                         min_samples_leaf=63, subsample=0.8, random_state=10, max_features=12)
modelfit(gbm_tuned_6) 


# In[14]:

model = GradientBoostingClassifier(learning_rate=0.05, loss='deviance',
              max_depth=4, max_features=18, max_leaf_nodes=None,
              min_samples_leaf=70, min_samples_split=2200,
              min_weight_fraction_leaf=0.0, n_estimators=2400,
              presort='auto', random_state=10, subsample=0.8, verbose=0,
              warm_start=False)


# In[15]:

model.fit(X_train, y_train)


# In[16]:

train_feat = model.predict_proba(X_train)[:,1]
valid_feat = model.predict_proba(X_valid)[:,1]
test_feat = model.predict_proba(X_test)[:,1]
train_feat = pd.DataFrame({"prob_gbdt":train_feat})
valid_feat = pd.DataFrame({"prob_gbdt":valid_feat})
test_feat = pd.DataFrame({"prob_gbdt":test_feat})
train_feat.to_csv("model/gbdt/prob_feat/train_feat.csv", index=False)
valid_feat.to_csv("model/gbdt/prob_feat/valid_feat.csv", index=False)
test_feat.to_csv("model/gbdt/prob_feat/test_feat.csv", index=False)


# In[116]:

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


# In[17]:

offline_train = pd.read_csv("train_dataset/train3.csv")
online_predict = pd.read_csv("predict_dataset/predict_4.csv")


# In[18]:

offline_X_train = offline_train.loc[:,"U_b1day_browse":].values
offline_y_train = offline_train.label.values
online_test = online_predict.loc[:,"U_b1day_browse":].values


# In[19]:

offline_X_train = min_max_scaler.transform(offline_X_train)
online_test = min_max_scaler.transform(online_test)


# In[20]:

model.fit(offline_X_train, offline_y_train)


# In[21]:

offline_train_prob = model.predict_proba(offline_X_train)[:,1]
offline_train_prob = pd.DataFrame({"prob_gbdt":offline_train_prob})
offline_train_prob.to_csv("model/gbdt/prob_feat/offline_train_feat.csv", index=False)


# In[ ]:

preds = model.predict_proba(online_test)[:,1]


# In[ ]:

preds = pd.DataFrame({"gbdt":preds})


# In[ ]:

preds.to_csv("model/gbdt/predict_prob.csv", index=False)


# In[68]:

index = preds.argsort()[-500:]


# In[69]:

submit_result = online_predict.loc[index]


# In[70]:

submit_result = submit_result[["user_id", "item_id"]]


# In[71]:

submit_result.to_csv("model/gbdt/tianchi_mobile_recommendation_predict.csv", index=False)


# In[ ]:



