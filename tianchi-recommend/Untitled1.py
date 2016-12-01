
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# sns.set_style('whitegrid')
# %matplotlib inline


# In[ ]:

all_dataset = pd.read_csv("tianchi_fresh_comp_train_user.csv")[['user_id','item_id','behavior_type','item_category','time']]
subset_item = pd.read_csv("tianchi_fresh_comp_train_item.csv")


# In[ ]:

##数据清洗
#去除不存在购买行为的用户
user_id_of_buy = all_dataset[all_dataset.behavior_type == 4]['user_id'].drop_duplicates()
user_id_of_buy = pd.DataFrame({'user_id':user_id_of_buy.values})
user_id_of_buy['flag'] = True
all_dataset = pd.merge(all_dataset,user_id_of_buy,on='user_id',how='left')
all_dataset = all_dataset[all_dataset.flag.notnull()].drop(['flag'], axis=1)


# In[ ]:

###去除购买能力低的，浏览/购买 < 500
buy_counts = all_dataset[all_dataset.behavior_type==4]['user_id'].value_counts()
browse_counts = all_dataset[all_dataset.behavior_type==1]['user_id'].value_counts()
buy_counts = pd.DataFrame({'user_id':buy_counts.index,'buy_counts':buy_counts.values})
browse_counts = pd.DataFrame({'user_id':browse_counts.index,'browse_counts':browse_counts.values})
buy_and_browse_counts = pd.merge(browse_counts,buy_counts,on='user_id')
normal_user = buy_and_browse_counts[buy_and_browse_counts.browse_counts / buy_and_browse_counts.buy_counts < 500]['user_id']
normal_user = pd.DataFrame({'user_id':normal_user.values})
normal_user['flag'] = True
all_dataset = pd.merge(all_dataset,normal_user,on='user_id',how='left')
all_dataset = all_dataset[all_dataset.flag.notnull()].drop('flag',axis=1)


# In[ ]:

###去除在对商品子集中无购买，收藏，加购物车的用户
subset_item_id = subset_item.item_id.drop_duplicates()
subset_item_id = pd.DataFrame({'item_id':subset_item_id.values})
subset_item_id['flag'] = True
all_dataset = pd.merge(all_dataset,subset_item_id,on='item_id',how='left')
subset_user_id = all_dataset[all_dataset.flag.notnull()]['user_id'].drop_duplicates()
all_dataset = all_dataset.drop(['flag'], axis=1)
subset_user_id = pd.DataFrame({'user_id':subset_user_id.values})
subset_user_id['flag'] = True
all_dataset = pd.merge(all_dataset,subset_user_id,on='user_id',how='left')
all_dataset = all_dataset[all_dataset.flag.notnull()].drop(['flag'], axis=1)


# In[ ]:

###去除只在双12购买商品的用户
double_12_buy_user_id = all_dataset[(all_dataset.time <= '2014-12-12 23') & (all_dataset.time >= '2014-12-12 0') & (all_dataset.behavior_type == 4)]['user_id'].drop_duplicates()
not_double_12_buy_user_id = all_dataset[((all_dataset.time > '2014-12-12 23') | (all_dataset.time < '2014-12-12 0')) & (all_dataset.behavior_type == 4)]['user_id'].drop_duplicates()
only_double_12_buy_user_id =pd.DataFrame({'user_id':list(set(double_12_buy_user_id) - set(not_double_12_buy_user_id))})
only_double_12_buy_user_id['flag'] = True
all_dataset = pd.merge(all_dataset,only_double_12_buy_user_id,on='user_id',how='left')
all_dataset = all_dataset[all_dataset.flag.isnull()].drop(['flag'], axis=1)


# In[ ]:

#训练集数据集
#训练集1
all_dataset_1 = all_dataset[(all_dataset.time >= '2014-11-18 0') & (all_dataset.time <= '2014-11-22 23')]
all_dataset_1_labelDay = all_dataset[(all_dataset.time >= '2014-11-23 0') & (all_dataset.time <= '2014-11-23 23')][['user_id','item_id','item_category','behavior_type']]
def buy_or_not(x):
    if x['behavior_type'] == 4:
        return 1
    else:
        return 0
all_dataset_1_labelDay['label'] = all_dataset_1_labelDay.apply(buy_or_not, axis=1)
all_dataset_1_labelDay = all_dataset_1_labelDay.drop('behavior_type', axis=1).drop_duplicates()


# In[15]:

#训练集2
all_dataset_2 = all_dataset[(all_dataset.time >= '2014-11-24 0') & (all_dataset.time <= '2014-11-28 23')]
all_dataset_2_labelDay = all_dataset[(all_dataset.time >= '2014-11-29 0') & (all_dataset.time <= '2014-11-29 23')][['user_id','item_id','item_category','behavior_type']]
def buy_or_not(x):
    if x['behavior_type'] == 4:
        return 1
    else:
        return 0
all_dataset_2_labelDay['label'] = all_dataset_2_labelDay.apply(buy_or_not, axis=1)
all_dataset_2_labelDay = all_dataset_2_labelDay.drop('behavior_type', axis=1).drop_duplicates()


# In[16]:

#训练集3
all_dataset_3 = all_dataset[(all_dataset.time >= '2014-11-30 0') & (all_dataset.time <= '2014-12-04 23')]
all_dataset_3_labelDay = all_dataset[(all_dataset.time >= '2014-12-05 0') & (all_dataset.time <= '2014-12-05 23')][['user_id','item_id','item_category','behavior_type']]
def buy_or_not(x):
    if x['behavior_type'] == 4:
        return 1
    else:
        return 0
all_dataset_3_labelDay['label'] = all_dataset_3_labelDay.apply(buy_or_not, axis=1)
all_dataset_3_labelDay = all_dataset_3_labelDay.drop('behavior_type', axis=1).drop_duplicates()


# In[17]:

#训练集4
all_dataset_4 = all_dataset[(all_dataset.time >= '2014-12-06 0') & (all_dataset.time <= '2014-12-10 23')]
all_dataset_4_labelDay = all_dataset[(all_dataset.time >= '2014-12-11 0') & (all_dataset.time <= '2014-12-11 23')][['user_id','item_id','item_category','behavior_type']]
def buy_or_not(x):
    if x['behavior_type'] == 4:
        return 1
    else:
        return 0
all_dataset_4_labelDay['label'] = all_dataset_4_labelDay.apply(buy_or_not, axis=1)
all_dataset_4_labelDay = all_dataset_4_labelDay.drop('behavior_type', axis=1).drop_duplicates()


# In[193]:

#线上测试集
all_dataset_5 = all_dataset[(all_dataset.time >= '2014-12-14 0') & (all_dataset.time <= '2014-12-18 23')]
all_dataset_5_labelDay = all_dataset_5[['user_id','item_id','item_category']]
subset_item = pd.read_csv("tianchi_fresh_comp_train_item.csv")
subset_item_id = subset_item[['item_id']].drop_duplicates()
subset_item_id['flag'] = True
all_dataset_5_labelDay = pd.merge(all_dataset_5_labelDay,subset_item_id,on='item_id',how='left')
all_dataset_5_labelDay = all_dataset_5_labelDay[all_dataset_5_labelDay['flag'] == True]
all_dataset_5_labelDay = all_dataset_5_labelDay.drop('flag', axis=1)



# In[ ]:

##特征工程


# In[20]:

def get_feat_UI(all_dataset_labelDay, all_dataset, start, end, name, flag):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset['behavior_type'] == flag)][['user_id','item_id']]
    temp = temp.groupby(['user_id', 'item_id']).size().reset_index()
    temp.rename(columns = {0: name}, inplace = True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay, temp, how='left', on=('user_id', 'item_id'))
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(0)
    return all_dataset_labelDay
    
def get_feat_UC(all_dataset_labelDay, all_dataset, start, end, name, flag):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset['behavior_type'] == flag)][['user_id','item_category']]
    temp = temp.groupby(['user_id', 'item_category']).size().reset_index()
    temp.rename(columns = {0: name}, inplace = True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay, temp, how='left', on=('user_id', 'item_category'))
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(0)
    return all_dataset_labelDay

def get_feat_U(all_dataset_labelDay, all_dataset, start, end, name, flag):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset['behavior_type'] == flag)][['user_id']]
    temp = temp.groupby(['user_id']).size().reset_index()
    temp.rename(columns = {0: name}, inplace = True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay, temp, how='left', on='user_id')
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(0)
    return all_dataset_labelDay

def get_feat_I(all_dataset_labelDay, all_dataset, start, end, name, flag):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset['behavior_type'] == flag)][['item_id']]
    temp = temp.groupby(['item_id']).size().reset_index()
    temp.rename(columns = {0: name}, inplace = True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay, temp, how='left', on='item_id')
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(0)
    return all_dataset_labelDay

def get_feat_C(all_dataset_labelDay, all_dataset, start, end, name, flag):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset['behavior_type'] == flag)][['item_category']]
    temp = temp.groupby(['item_category']).size().reset_index()
    temp.rename(columns = {0: name}, inplace = True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay, temp, how='left', on='item_category')
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(0)
    return all_dataset_labelDay


# In[21]:

from datetime import datetime
#获取用户特征--加购到购买的平均时间间隔
cart_dataset = all_dataset[all_dataset.behavior_type == 3][['user_id', 'item_id', 'time']]
cart_dataset.columns = ['user_id','item_id','cart_time']
cart_dataset = cart_dataset.drop_duplicates()
buy_dataset = all_dataset[all_dataset.behavior_type == 4][['user_id', 'item_id', 'time']]
buy_dataset.columns = ['user_id','item_id','buy_time']
buy_dataset = buy_dataset.drop_duplicates()
buy_dataset = buy_dataset.sort(['buy_time'])
cart_dataset = cart_dataset.sort(['cart_time'])
print len(cart_dataset)
CB_dataset = pd.merge(cart_dataset,buy_dataset,on=('user_id','item_id'),how='left')
print len(CB_dataset)
CB_dataset = CB_dataset.dropna()
def calc_time(x):
    time_step = (datetime.strptime(x['buy_time'],'%Y-%m-%d %H') - datetime.strptime(x['cart_time'],'%Y-%m-%d %H')).total_seconds() / 3600
    if time_step >= 144:
        time_step = 144.0
    elif time_step < 0:
        #预防出现购买出现在加购物车之前
        time_step = 0.0
    return time_step
CB_dataset['time'] = CB_dataset.apply(calc_time,axis=1)
CB_dataset = CB_dataset[['user_id','time']]
mean_time = CB_dataset.time.groupby(CB_dataset.user_id).mean()
mean_time = pd.DataFrame({'user_id':mean_time.index,'mean_time':mean_time.values})
CB_dataset = pd.merge(CB_dataset,mean_time,on='user_id',how='left').drop_duplicates()
CB_dataset = CB_dataset.drop('time', axis=1)
CB_dataset = CB_dataset.drop_duplicates()

def get_feature_mean_time(train_item, CB_dataset = CB_dataset):
    train_item = pd.merge(train_item, CB_dataset, on='user_id', how='left')
    train_item['mean_time'] = train_item['mean_time'].fillna(24.0)
    return train_item

def calc_near(x):
    time_step = (datetime.strptime(x['label_time'],'%Y-%m-%d %H') - datetime.strptime(x['time'],'%Y-%m-%d %H')).total_seconds() / 3600
    return time_step
#获取最后某种操作到Label日的时间
def get_feature_near_time(train_item, all_dataset, start, end, name, flag, label_time):
    temp = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset.behavior_type == flag)][['user_id','item_id','time']]
    temp['label_time'] = label_time
    temp[name] = temp.apply(calc_near,axis=1)
    ##排序后删除重复值，会默认保留第一个值，可以达到获取最小值的目的
    temp = temp.drop_duplicates().sort(name)
    temp = temp.loc[temp[['user_id','item_id']].drop_duplicates().index][['user_id','item_id',name]]
    train_item = pd.merge(train_item,temp,on=('user_id','item_id'),how='left')
    train_item[name] = train_item[name].fillna(train_item[name].max())
    return train_item


# In[22]:

##获取用户感兴趣的商品(按浏览)
def get_interest(all_dataset_labelDay, all_dataset, name, flag):
    a = all_dataset[(all_dataset.behavior_type == flag)][['user_id','item_id']]
    a = a.groupby(['user_id','item_id']).size().reset_index()
    a.rename(columns = {0: 'count'}, inplace = True)
    b = a[['user_id','count']].groupby('user_id').max().reset_index()
    b.rename(columns = {'count': 'max_count'}, inplace = True)
    a[['count']] = a[['count']].astype(float)
    a = pd.merge(a,b,on='user_id',how='left')
    a[name] = a['count'] / a['max_count']
    a = a.drop(['count','max_count'], axis=1)
    a = pd.DataFrame(a)
    a = a.drop_duplicates()
    all_dataset_labelDay = pd.merge(all_dataset_labelDay,a,on=('user_id','item_id'),how='left')
    all_dataset_labelDay[name] = all_dataset_labelDay[name].fillna(1e-5)
    return all_dataset_labelDay


# In[ ]:




# In[23]:

##获取用户在一段时间内与其他类商品具有操作（如是否加购同类的其他商品,购买同类的其他商品）
def get_feat_on_other_c(all_dataset_labelDay, all_dataset, start, end, flag, name):
    a = all_dataset[(all_dataset.time >= start) & (all_dataset.time <= end) & (all_dataset.behavior_type == flag)][['user_id','item_id','item_category']]
    a = a.drop_duplicates()
    a['flag1'] = 1
    all_dataset_labelDay = pd.merge(all_dataset_labelDay,a,on=('user_id','item_id','item_category'),how = 'left')
    all_dataset_labelDay['flag1'] = all_dataset_labelDay['flag1'].fillna(0)
    a = a.drop(['item_id','flag1'], axis=1)
    a = a.groupby(['user_id','item_category']).size().reset_index()
    a.rename(columns={0:'flag2'},inplace=True)
    all_dataset_labelDay = pd.merge(all_dataset_labelDay,a,on=('user_id','item_category'),how='left')
    all_dataset_labelDay['flag2'] = all_dataset_labelDay['flag2'].fillna(0)
    all_dataset_labelDay[name] = all_dataset_labelDay['flag2'] - all_dataset_labelDay['flag1']
    all_dataset_labelDay[name] = all_dataset_labelDay.apply(lambda x:1 if x[name] > 0 else 0,axis=1)
    all_dataset_labelDay = all_dataset_labelDay.drop(['flag1','flag2'], axis=1)
    return all_dataset_labelDay


# In[24]:

###开始特征获取


# In[25]:

#训练集一
##UI特征
def get_finally_feat(all_dataset_labelDay, all_dataset_part, all_dataset, train_ID):
    time_lists = []
    time_lists.append(['2014-11-22 20','2014-11-22 16','2014-11-22 08','2014-11-22 00','2014-11-20 00','2014-11-18 00','2014-11-23 00','2014-11-22 23'])
    time_lists.append(['2014-11-28 20','2014-11-28 16','2014-11-28 08','2014-11-28 00','2014-11-26 00','2014-11-24 00','2014-11-29 00','2014-11-28 23'])
    time_lists.append(['2014-12-04 20','2014-12-04 16','2014-12-04 08','2014-12-04 00','2014-12-02 00','2014-11-30 00','2014-12-05 00','2014-12-04 23'])
    time_lists.append(['2014-12-10 20','2014-12-10 16','2014-12-10 08','2014-12-10 00','2014-12-08 00','2014-12-06 00','2014-12-11 00','2014-12-10 23'])
    time_lists.append(['2014-12-18 20','2014-12-18 16','2014-12-18 08','2014-12-18 00','2014-12-16 00','2014-12-14 00','2014-12-19 00','2014-12-18 23'])
    time_list = time_lists[train_ID]
    ##UI特征
    X1 = get_feat_UI(all_dataset_labelDay, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_bro_UI', flag=1)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_bro_UI', flag=1)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_bro_UI', flag=1)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_bro_UI', flag=1)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_bro_UI', flag=1)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_bro_UI', flag=1)

    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_col_UI', flag=2)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_col_UI', flag=2)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_col_UI', flag=2)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_col_UI', flag=2)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_col_UI', flag=2)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_col_UI', flag=2)

    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_cart_UI', flag=3)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_cart_UI', flag=3)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_cart_UI', flag=3)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_cart_UI', flag=3)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_cart_UI', flag=3)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_cart_UI', flag=3)
    
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_buy_UI', flag=4)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_buy_UI', flag=4)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_buy_UI', flag=4)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_buy_UI', flag=4)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_buy_UI', flag=4)
    X1 = get_feat_UI(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_buy_UI', flag=4)
    
    ##UC特征
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_bro_UC', flag=1)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_bro_UC', flag=1)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_bro_UC', flag=1)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_bro_UC', flag=1)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_bro_UC', flag=1)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_bro_UC', flag=1)

    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_col_UC', flag=2)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_col_UC', flag=2)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_col_UC', flag=2)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_col_UC', flag=2)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_col_UC', flag=2)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_col_UC', flag=2)

    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_cart_UC', flag=3)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_cart_UC', flag=3)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_cart_UC', flag=3)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_cart_UC', flag=3)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_cart_UC', flag=3)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_cart_UC', flag=3)
    
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_buy_UC', flag=4)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_buy_UC', flag=4)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_buy_UC', flag=4)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_buy_UC', flag=4)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_buy_UC', flag=4)
    X1 = get_feat_UC(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_buy_UC', flag=4)
    
    ##U特征
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_bro_U', flag=1)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_bro_U', flag=1)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_bro_U', flag=1)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_bro_U', flag=1)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_bro_U', flag=1)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_bro_U', flag=1)

    X1 = get_feat_U(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_col_U', flag=2)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_col_U', flag=2)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_col_U', flag=2)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_col_U', flag=2)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_col_U', flag=2)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_col_U', flag=2)

    X1 = get_feat_U(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_cart_U', flag=3)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_cart_U', flag=3)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_cart_U', flag=3)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_cart_U', flag=3)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_cart_U', flag=3)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_cart_U', flag=3)
    
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_buy_U', flag=4)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_buy_U', flag=4)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_buy_U', flag=4)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_buy_U', flag=4)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_buy_U', flag=4)
    X1 = get_feat_U(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_buy_U', flag=4)
    
    ##I特征
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_bro_I', flag=1)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_bro_I', flag=1)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_bro_I', flag=1)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_bro_I', flag=1)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_bro_I', flag=1)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_bro_I', flag=1)

    X1 = get_feat_I(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_col_I', flag=2)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_col_I', flag=2)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_col_I', flag=2)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_col_I', flag=2)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_col_I', flag=2)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_col_I', flag=2)

    X1 = get_feat_I(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_cart_I', flag=3)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_cart_I', flag=3)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_cart_I', flag=3)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_cart_I', flag=3)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_cart_I', flag=3)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_cart_I', flag=3)
    
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_buy_I', flag=4)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_buy_I', flag=4)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_buy_I', flag=4)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_buy_I', flag=4)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_buy_I', flag=4)
    X1 = get_feat_I(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_buy_I', flag=4)
    
    ##C特征
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_bro_C', flag=1)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_bro_C', flag=1)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_bro_C', flag=1)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_bro_C', flag=1)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_bro_C', flag=1)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_bro_C', flag=1)

    X1 = get_feat_C(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_col_C', flag=2)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_col_C', flag=2)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_col_C', flag=2)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_col_C', flag=2)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_col_C', flag=2)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_col_C', flag=2)

    X1 = get_feat_C(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_cart_C', flag=3)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_cart_C', flag=3)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_cart_C', flag=3)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_cart_C', flag=3)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_cart_C', flag=3)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_cart_C', flag=3)
    
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[0], end=time_list[-1], name='b_4_h_buy_C', flag=4)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[1], end=time_list[-1], name='b_8_h_buy_C', flag=4)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[2], end=time_list[-1], name='b_16_h_buy_C', flag=4)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[3], end=time_list[-1], name='b_1_d_buy_C', flag=4)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[4], end=time_list[-1], name='b_3_d_buy_C', flag=4)
    X1 = get_feat_C(X1, all_dataset_part, start=time_list[5], end=time_list[-1], name='b_5_d_buy_C', flag=4)
    
    ##用户加购物车到购买商品的平均时间(全集)
    X1 = get_feature_mean_time(X1)
    
    ## 用户最近一次加购物车到label日的时间(全集)
    X1 = get_feature_near_time(X1,all_dataset,start=time_list[0],end=time_list[-1],name='cart_near_time',flag=3,label_time=time_list[-2])
    
    ##  用户最近一次购买该商品到labels日的时间(全集)
    X1 = get_feature_near_time(X1,all_dataset,start=time_list[0],end=time_list[-1],name='buy_near_time',flag=4,label_time=time_list[-2])
    
    ##用户对商品的喜爱程度1(按照浏览次数排序)(全集) over
    X1 = get_interest(X1,all_dataset,name='interest_1',flag=1)
    
    ##用户对商品的喜爱程度2(按照购买次数排序)(全集) over
    X1 = get_interest(X1,all_dataset,name='interest_2',flag=4)
    
    ##用户在考察时间段是否加够其他同类的商品 over
    X1 = get_feat_on_other_c(X1,all_dataset,start=time_list[-3],end=time_list[-1],flag=3,name='cart_other')
    
    ##用户在考察时间段是否加够其他同类的商品 over
    X1 = get_feat_on_other_c(X1,all_dataset,start=time_list[-3],end=time_list[-1],flag=4,name='buy_other')   
    
    ##交叉特征
    ##UI浏览-购买转化率:每个时间段的购买数/浏览数(含全集)
    X1 = get_feat_UI(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_bro_UI',flag=1)
    X1 = get_feat_UI(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_col_UI',flag=2)
    X1 = get_feat_UI(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_cart_UI',flag=3)
    X1 = get_feat_UI(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_buy_UI',flag=4)
    X1['convert_1'] = (X1['all_buy_UI'] + 1e-6) / (X1['all_bro_UI'] + 1e-5)
    X1['convert_2'] = (X1['all_buy_UI'] + 1e-6) / (X1['all_col_UI'] + 1e-5)
    X1['convert_3'] = (X1['all_buy_UI'] + 1e-6) / (X1['all_cart_UI'] + 1e-5)
    X1 = X1.drop(['all_bro_UI','all_col_UI','all_cart_UI','all_buy_UI'], axis=1)
    
    ##U的3种操作-购买转化率(全集)
    X1 = get_feat_U(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_bro_U',flag=1)
    X1 = get_feat_U(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_col_U',flag=2)
    X1 = get_feat_U(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_cart_U',flag=3)
    X1 = get_feat_U(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_buy_U',flag=4)
    X1['convert_4'] = (X1['all_buy_U'] + 1e-6) / (X1['all_bro_U'] + 1e-5)
    X1['convert_5'] = (X1['all_buy_U'] + 1e-6) / (X1['all_col_U'] + 1e-5)
    X1['convert_6'] = (X1['all_buy_U'] + 1e-6) / (X1['all_cart_U'] + 1e-5)
    X1 = X1.drop(['all_bro_U','all_col_U','all_cart_U','all_buy_U'], axis=1)
    
    ##I的3种操作-购买转化率(全集)
    X1 = get_feat_I(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_bro_I',flag=1)
    X1 = get_feat_I(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_col_I',flag=2)
    X1 = get_feat_I(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_cart_I',flag=3)
    X1 = get_feat_I(X1,all_dataset,start='2014-11-18 0',end='2014-12-18 23',name='all_buy_I',flag=4)
    X1['convert_7'] = (X1['all_buy_I'] + 1e-6) / (X1['all_bro_I'] + 1e-5)
    X1['convert_8'] = (X1['all_buy_I'] + 1e-6) / (X1['all_col_I'] + 1e-5)
    X1['convert_9'] = (X1['all_buy_I'] + 1e-6) / (X1['all_cart_I'] + 1e-5)
    X1 = X1.drop(['all_bro_I','all_col_I','all_cart_I','all_buy_I'], axis=1)
    
    return X1


# In[ ]:

X1 = get_finally_feat(all_dataset_1_labelDay,all_dataset_1,all_dataset,0)


# In[ ]:

X2 = get_finally_feat(all_dataset_2_labelDay,all_dataset_2,all_dataset,1)


# In[ ]:

X3 = get_finally_feat(all_dataset_3_labelDay,all_dataset_3,all_dataset,2)


# In[ ]:

X4 = get_finally_feat(all_dataset_4_labelDay,all_dataset_4,all_dataset,3)


# In[ ]:

X5 = get_finally_feat(all_dataset_5_labelDay,all_dataset_5,all_dataset,4)


# In[ ]:

# In[257]:

from sklearn.preprocessing import scale
from random import sample


# In[258]:

def get_random_train_X(X):
    right_len = len(X[X.label == 1])
    item_index = sample(X[X['label'] == 0].index, 10*right_len)
    item_index.extend(X[X['label'] == 1].index)
    item_index = sample(item_index, len(item_index))
    train_item_temp = X.loc[item_index]
    train_item_temp.index = range(len(train_item_temp))
    return train_item_temp


# In[259]:

random_X1 = get_random_train_X(X1)
train_X1 = scale(random_X1.loc[:,'b_4_h_bro_UI':])
train_Y1 = random_X1['label'].values


# In[260]:

random_X2 = get_random_train_X(X2)
train_X2 = scale(random_X2.loc[:,'b_4_h_bro_UI':])
train_Y2 = random_X2['label'].values


# In[261]:

random_X3 = get_random_train_X(X3)
train_X3 = scale(random_X3.loc[:,'b_4_h_bro_UI':])
train_Y3 = random_X3['label'].values


# In[262]:

# random_X4 = get_random_train_X(X4)
# train_X4 = scale(random_X4.loc[:,'b_4_h_bro_UI':])
# train_Y4 = random_X4['label'].values


# In[285]:

valid_X4 = scale(X4.loc[:,'b_4_h_bro_UI':])


# In[286]:

train_X5 = scale(X5.loc[:,'b_4_h_bro_UI':])


# In[ ]:

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(min_samples_split=3,n_estimators=1000,learning_rate=0.04)


# In[271]:

model.fit(np.concatenate((train_X1,train_X2,train_X3)),np.concatenate((train_Y1,train_Y2,train_Y3)))


# In[272]:




# In[273]:




# In[274]:




# In[312]:

def get_accuary(X,valid_X):
    item_index = model.predict_proba(valid_X)[:,1].argsort()[-8000:]
    subset_item = pd.read_csv("tianchi_fresh_comp_train_item.csv")
    subset_item_id = subset_item[['item_id']].drop_duplicates()
    subset_item_id['flag'] = True
    # Y_pre = pd.DataFrame({'label':Y_pre})
        #预测被购买的商品
    submit_item = X.loc[item_index][['user_id','item_id']]
        ##与subset取交集
    submit_item = pd.merge(submit_item, subset_item_id, on='item_id', how='left')
    submit_item = submit_item[submit_item['flag'] == True]
    submit_item = submit_item.drop('flag', axis=1)
    
#     print len(submit_item)
    real_item = X.loc[X[X.label == 1].index][['user_id','item_id']]
    real_item = pd.merge(real_item, subset_item_id, on='item_id', how='left')
    real_item = real_item[real_item['flag'] == True]
    real_item = real_item.drop('flag', axis=1)
#     print len(real_item)
    
    submit_set = set()
    for value in submit_item.itertuples():
        submit_set.add((value[1],value[2]))
    real_set = set()
    for value in real_item.itertuples():
        real_set.add((value[1],value[2]))
    
    print len(submit_set)
    print len(real_set)
    p = len(submit_set & real_set) / float(len(submit_set))
    print p
    r = len(submit_set & real_set) / float(len(real_set))
    print r
    f1 = (2 * p * r) / (p + r)
    print f1


# In[311]:

get_accuary(X4,valid_X4)


# In[ ]:

item_index = model.predict_proba(train_X5)[:,1].argsort()[-8000:]
subset_item = pd.read_csv("tianchi_fresh_comp_train_item.csv")
subset_item_id = subset_item[['item_id']].drop_duplicates()
subset_item_id['flag'] = True
# Y_pre = pd.DataFrame({'label':Y_pre})
    #预测被购买的商品
submit_item = X5.loc[item_index][['user_id','item_id']]
    ##与subset取交集
submit_item = pd.merge(submit_item, subset_item_id, on='item_id', how='left')
submit_item = submit_item[submit_item['flag'] == True]
submit_item = submit_item.drop('flag', axis=1)
submit_item = submit_item.drop_duplicates()


# In[319]:

print len(submit_item)



# In[ ]:

# In[320]:

submit_item.to_csv("tianchi_mobile_recommendation_predict.csv", index=False)


# In[315]:

from sklearn.externals import joblib
joblib.dump(model, 'GDBT_model/GDBT.pkl')


# In[ ]:



