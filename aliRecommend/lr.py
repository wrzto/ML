
# coding: utf-8

# In[47]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[48]:

subset = pd.read_csv("tianchi_fresh_comp_train_item.csv")


# In[49]:

train = pd.read_csv("train_dataset/train1.csv")
valid = pd.read_csv("predict_dataset/predict_2.csv")
test = pd.read_csv("predict_dataset/predict_3.csv")
# valid = pd.read_csv("train_dataset/train2.csv")
# test = pd.read_csv("train_dataset/train3.csv")


# In[50]:

valid_num = len(pd.read_csv("label_dataset/labelDataset_2.csv"))
test_num = len(pd.read_csv("label_dataset/labelDataset_3.csv"))


# In[51]:

# import pickle
# with open("model/xgb/feature_top46.pkl", "rb") as f:
#     important_feature = pickle.load(f)


# In[52]:

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


# In[53]:

X_train = train.loc[:,"U_b1day_browse":].values
# X_train = train[important_feature].values
y_train = train.label.values


# In[54]:

X_valid = valid.loc[:,"U_b1day_browse":].values
# X_valid = valid[important_feature].values
y_valid = valid.label.values


# In[55]:

X_test = test.loc[:,"U_b1day_browse":].values
# X_test = test[important_feature].values
y_test = test.label.values


# In[56]:

min_max_scaler = MinMaxScaler()


# In[57]:

X_train = min_max_scaler.fit_transform(X_train)
X_valid = min_max_scaler.transform(X_valid)
X_test = min_max_scaler.transform(X_test)


# In[58]:

valid = valid[["user_id", "item_id", "label"]]
test = test[["user_id", "item_id", "label"]]


# In[59]:

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[60]:

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


# In[61]:

model = LogisticRegression()
modelfit(model)


# In[71]:

param_test1 = {"C":[2,3,4]}
gsearch1 = GridSearchCV(estimator = LogisticRegression(),
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)


# In[72]:

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[101]:

model = LogisticRegression(C=2)
model.fit(X_train, y_train)


# In[103]:

train_feat = model.predict_proba(X_train)[:,1]
valid_feat = model.predict_proba(X_valid)[:,1]
test_feat = model.predict_proba(X_test)[:,1]
train_feat = pd.DataFrame({"prob_lr":train_feat})
valid_feat = pd.DataFrame({"prob_lr":valid_feat})
test_feat = pd.DataFrame({"prob_lr":test_feat})
train_feat.to_csv("model/lr/prob_feat/train_feat.csv", index=False)
valid_feat.to_csv("model/lr/prob_feat/valid_feat.csv", index=False)
test_feat.to_csv("model/lr/prob_feat/test_feat.csv", index=False)


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


# In[106]:

offline_train = pd.read_csv("train_dataset/train3.csv")
online_predict = pd.read_csv("predict_dataset/predict_4.csv")


# In[107]:

offline_X_train = offline_train.loc[:,"U_b1day_browse":].values
offline_y_train = offline_train.label.values
online_test = online_predict.loc[:,"U_b1day_browse":].values


# In[108]:

offline_X_train = min_max_scaler.transform(offline_X_train)
online_test = min_max_scaler.transform(online_test)


# In[109]:

model.fit(offline_X_train, offline_y_train)


# In[112]:

offline_train_prob = model.predict_proba(offline_X_train)[:,1]


# In[113]:

offline_train_prob = pd.DataFrame({"prob_lr":offline_train_prob})


# In[115]:

offline_train_prob.to_csv("model/lr/prob_feat/offline_train_feat.csv", index=False)


# In[81]:

preds = model.predict_proba(online_test)[:,1]


# In[82]:

preds = pd.DataFrame({"lr":preds})


# In[83]:

preds.to_csv("model/lr/predict_prob.csv", index=False)


# In[68]:

index = preds.argsort()[-500:]


# In[69]:

submit_result = online_predict.loc[index]


# In[70]:

submit_result = submit_result[["user_id", "item_id"]]


# In[71]:

submit_result.to_csv("model/gbdt/tianchi_mobile_recommendation_predict.csv", index=False)


# In[ ]:



