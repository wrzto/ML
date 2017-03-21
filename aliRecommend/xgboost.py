
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


# In[4]:

valid_num = len(pd.read_csv("label_dataset/labelDataset_2.csv"))
test_num = len(pd.read_csv("label_dataset/labelDataset_3.csv"))


# In[5]:

X_train = train.loc[:,"U_b1day_browse":].values
y_train = train.label.values


# In[6]:

X_valid = valid.loc[:,"U_b1day_browse":].values
y_valid = valid.label.values


# In[7]:

X_test = test.loc[:,"U_b1day_browse":].values
y_test = test.label.values


# In[8]:

min_max_scaler = MinMaxScaler()


# In[9]:

X_train = min_max_scaler.fit_transform(X_train)
X_valid = min_max_scaler.transform(X_valid)
X_test = min_max_scaler.transform(X_test)


# In[10]:

valid = valid[["user_id", "item_id", "label"]]
test = test[["user_id", "item_id", "label"]]


# In[11]:

import xgboost as xgb


# In[12]:

dtrain = xgb.DMatrix(data=X_train, label=y_train)


# In[13]:

dtest = xgb.DMatrix(data=X_test)


# In[14]:

dvalid = xgb.DMatrix(data=X_valid)


# In[15]:

param = {"colsample_bytree":0.8, "gamma":0.7, "learning_rate":0.01, "max_depth":5, "min_child_weight":1, "objective":'binary:logistic',
         "reg_alpha":1e-05, "subsample":0.8, "silent":True}


# In[16]:

num_round = 1614
model = xgb.train(param, dtrain, num_round)


# In[18]:

train_feat = model.predict(dtrain)
valid_feat = model.predict(dvalid)
test_feat = model.predict(dtest)
train_feat = pd.DataFrame({"prob_xgb":train_feat})
valid_feat = pd.DataFrame({"prob_xgb":valid_feat})
test_feat = pd.DataFrame({"prob_xgb":test_feat})
train_feat.to_csv("model/xgb/prob_feat/train_feat.csv", index=False)
valid_feat.to_csv("model/xgb/prob_feat/valid_feat.csv", index=False)
test_feat.to_csv("model/xgb/prob_feat/test_feat.csv", index=False)


# In[19]:

preds = model.predict(dvalid)
index = preds.argsort()[-500:]
preds = valid.loc[index]
preds = preds[preds.item_id.isin(subset.item_id)]
p = len(preds[preds.label == 1]) / len(preds)
r = len(preds[preds.label == 1]) / valid_num
print(len(preds[preds.label == 1]))
print((2 * p * r) / (p + r))


preds = model.predict(dtest)
index = preds.argsort()[-500:]
preds = test.loc[index]
preds = preds[preds.item_id.isin(subset.item_id)]
p = len(preds[preds.label == 1]) / len(preds)
r = len(preds[preds.label == 1]) / test_num
print(len(preds[preds.label == 1]))
print((2 * p * r) / (p + r))


# In[196]:

feature_score = sorted(model.get_score().items(),key=lambda x:x[-1], reverse=True)


# In[197]:

important_feature = list(filter(lambda x: x[-1]>= 300,feature_score))


# In[199]:

important_feature = list(map(lambda x:int(x[0][1:]), important_feature))


# In[202]:

important_feature = list(online_predict.columns[3:][important_feature].values)


# In[203]:

import pickle
with open("model/xgb/feature_top46.pkl", "wb") as f:
    pickle.dump(important_feature, f)


# In[20]:

offline_train = pd.read_csv("train_dataset/train3.csv")
online_predict = pd.read_csv("predict_dataset/predict_4.csv")


# In[21]:

offline_X_train = offline_train.loc[:,"U_b1day_browse":].values
offline_y_train = offline_train.label.values
online_test = online_predict.loc[:,"U_b1day_browse":].values


# In[22]:

offline_X_train = min_max_scaler.transform(offline_X_train)
online_test = min_max_scaler.transform(online_test)


# In[ ]:

offline_dtrain = xgb.DMatrix(data=offline_X_train, label=offline_y_train)
online_dtest = xgb.DMatrix(data=online_test)


# In[ ]:

param = {"colsample_bytree":0.8, "gamma":0.7, "learning_rate":0.01, "max_depth":5, "min_child_weight":1, "objective":'binary:logistic',
         "reg_alpha":1e-05, "subsample":0.8, "silent":True}
num_round = 1614
model = xgb.train(param, offline_dtrain, num_round)


# In[ ]:

offline_train_prob = model.predict(offline_X_dtrain)
offline_train_prob = pd.DataFrame({"prob_xgb":offline_train_prob})
offline_train_prob.to_csv("model/xgb/prob_feat/offline_train_feat.csv", index=False)


# In[23]:

preds = model.predict(online_dtest)


# In[25]:

preds = pd.DataFrame({'xgboost':preds})


# In[27]:

preds.to_csv("model/xgb/predict_prob.csv", index=False)


# In[79]:

index = preds.argsort()[-500:]


# In[81]:

submit_result = online_predict.loc[index]


# In[83]:

submit_result = submit_result[["user_id", "item_id"]]


# In[87]:

submit_result.to_csv("model/xgb/tianchi_mobile_recommendation_predict.csv", index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



