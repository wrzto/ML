
# coding: utf-8

# In[1]:

import pandas as pd


# In[4]:

# subset_dataset = pd.read_csv("tianchi_fresh_comp_train_item.csv")
all_dataset = pd.read_csv("tianchi_fresh_comp_train_user.csv")


# In[5]:

##去除爬虫用户 (浏览+收藏+加购) > 500
users = all_dataset[["user_id"]].drop_duplicates()


# In[14]:

#统计统计用户一个月的浏览数
users_total_browse = all_dataset[all_dataset.behavior_type == 1].groupby("user_id").size().reset_index()
users_total_browse.rename(columns={0:"total_month_browse"}, inplace=True)


# In[16]:

#统计统计用户一个月的收藏数
users_total_collect = all_dataset[all_dataset.behavior_type == 2].groupby("user_id").size().reset_index()
users_total_collect.rename(columns={0:"total_month_collect"}, inplace=True)


# In[18]:

#统计统计用户一个月的加购数
users_total_cart = all_dataset[all_dataset.behavior_type == 3].groupby("user_id").size().reset_index()
users_total_cart.rename(columns={0:"total_month_cart"}, inplace=True)


# In[20]:

#统计用户一个月的购买数
users_total_buy = all_dataset[all_dataset.behavior_type == 4].groupby("user_id").size().reset_index()
users_total_buy.rename(columns={0:"total_month_buy"}, inplace=True)


# In[22]:

users = pd.merge(users, users_total_browse, on="user_id", how="left")
users = pd.merge(users, users_total_collect, on="user_id", how="left")
users = pd.merge(users, users_total_cart, on="user_id", how="left")
users = pd.merge(users, users_total_buy, on="user_id", how="left")


# In[24]:

users.fillna(0, inplace=True)


# In[29]:

users = users[users.total_month_buy > 0]


# In[42]:

users = users[users.loc[:,"total_month_browse":"total_month_cart"].sum(axis=1) / users.total_month_buy < 500]


# In[43]:

all_dataset = all_dataset[all_dataset.user_id.isin(users.user_id)]


# In[45]:

#split dataset
##11-17-->12-15提取特征用作训练集
##11-17-->12-16提取特征用作测试集
##11-17-->12-17提取特征用作验证集
##测试集和验证集f1差距用来判断模型是否有用
##11-17-->12-17提取特征用作线上训练集
##11-17-->12-18提取特征用线上测试集作集
extract_feat_dataset_15 = all_dataset[(all_dataset.time <= "2014-12-15 23")]
extract_feat_dataset_16 = all_dataset[(all_dataset.time <= "2014-12-16 23")]
extract_feat_dataset_17 = all_dataset[(all_dataset.time <= "2014-12-17 23")]
extract_feat_dataset_18 = all_dataset[(all_dataset.time <= "2014-12-18 23")]


# In[48]:

label = all_dataset[(all_dataset.time.str.startswith("2014-12-16")) & (all_dataset.behavior_type == 4)][["user_id", "item_id"]] 


# In[52]:

label = label.drop_duplicates()
label["label"] = 1


# In[54]:

extract_feat_dataset_15 = pd.merge(extract_feat_dataset_15, label, on=("user_id", "item_id"), how="left")


# In[56]:

extract_feat_dataset_15.label = extract_feat_dataset_15.label.fillna(0)


# In[61]:

extract_feat_dataset_15.to_csv("extract_dataset/extract_feat_dataset_15.csv", index=False)


# In[62]:

label = all_dataset[(all_dataset.time.str.startswith("2014-12-17")) & (all_dataset.behavior_type == 4)][["user_id", "item_id"]] 
label = label.drop_duplicates()
label["label"] = 1
extract_feat_dataset_16 = pd.merge(extract_feat_dataset_16, label, on=("user_id", "item_id"), how="left")
extract_feat_dataset_16.label = extract_feat_dataset_16.label.fillna(0)
extract_feat_dataset_16.to_csv("extract_dataset/extract_feat_dataset_16.csv", index=False)


# In[63]:

label = all_dataset[(all_dataset.time.str.startswith("2014-12-18")) & (all_dataset.behavior_type == 4)][["user_id", "item_id"]] 
label = label.drop_duplicates()
label["label"] = 1
extract_feat_dataset_17 = pd.merge(extract_feat_dataset_17, label, on=("user_id", "item_id"), how="left")
extract_feat_dataset_17.label = extract_feat_dataset_17.label.fillna(0)
extract_feat_dataset_17.to_csv("extract_dataset/extract_feat_dataset_17.csv", index=False)


# In[64]:

extract_feat_dataset_18.to_csv("extract_dataset/extract_feat_dataset_18.csv", index=False)


# In[ ]:



