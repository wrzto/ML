# coding: utf-8

import pandas as pd
import numpy as np

##统计用户特征 4种行为数 / 时间
def get_feat_U(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["user_id"]]
    temp = temp.groupby("user_id").size().reset_index().rename(columns={0:column_name})
    train_item = pd.merge(train_item, temp, on=("user_id"), how = 'left')
    train_item[column_name] = train_item[column_name].fillna(0)
    TimeDelta = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24*60*60)
    train_item[column_name] /= TimeDelta
    return train_item

def get_feat_I(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["item_id"]]
    temp = temp.groupby("item_id").size().reset_index().rename(columns={0:column_name})
    train_item = pd.merge(train_item, temp, on=("item_id"), how = 'left')
    train_item[column_name] = train_item[column_name].fillna(0)
    TimeDelta = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24*60*60)
    train_item[column_name] /= TimeDelta
    return train_item

def get_feat_C(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["item_category"]]
    temp = temp.groupby("item_category").size().reset_index().rename(columns={0:column_name})
    train_item = pd.merge(train_item, temp, on=("item_category"), how = 'left')
    train_item[column_name] = train_item[column_name].fillna(0)
    TimeDelta = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24*60*60)
    train_item[column_name] /= TimeDelta
    return train_item


###获取是否对同类其他商品具有某种操作
def get_feat_other_i(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["user_id","item_id","item_category"]].drop_duplicates()
    temp1 = temp.groupby(["user_id","item_category"]).size().reset_index().rename(columns={0:column_name})
    temp1[column_name] = temp1.apply(lambda x:1 if x[column_name] > 1 else 0, axis=1)
    temp = pd.merge(temp ,temp1, on=("user_id","item_category"), how="left")
    temp.drop("item_category", axis=1, inplace=True)
    temp.fillna(0, inplace=True)
    temp = temp.drop_duplicates()
    train_item = pd.merge(train_item, temp, on=("user_id","item_id"), how="left")
    train_item[column_name] = train_item[column_name].fillna(0)
    return train_item

##feat获取UI,UC的各种操作计数
def get_feat_UI(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["user_id","item_id"]]
    temp = temp.groupby(["user_id","item_id"]).size().reset_index().rename(columns={0:column_name})
    train_item = pd.merge(train_item, temp, on=("user_id","item_id"), how = 'left')
    train_item[column_name] = train_item[column_name].fillna(0)
    TimeDelta = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24*60*60)
    train_item[column_name] /= TimeDelta
    return train_item

def get_feat_UC(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["user_id","item_category"]]
    temp = temp.groupby(["user_id","item_category"]).size().reset_index().rename(columns={0:column_name})
    train_item = pd.merge(train_item, temp, on=("user_id","item_category"), how = 'left')
    train_item[column_name] = train_item[column_name].fillna(0)
    TimeDelta = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24*60*60)
    train_item[column_name] /= TimeDelta
    return train_item

###feat获取是否对商品进行过某种操作
def get_feat_or_not(train_item, all_data, start, end, flag, column_name):
    temp = all_data[(all_data.time >= start) & (all_data.time < end) & (all_data.behavior_type == flag)][["user_id","item_id"]].drop_duplicates()
    temp[column_name] = 1
    train_item = pd.merge(train_item, temp, on=("user_id","item_id"), how="left")
    train_item[column_name] = train_item[column_name].fillna(0)
    return train_item

def extract_feat(feat, dataset, t1, t2, t3, t4, t5, t6):
    ###用户特征
    ##前1天
    feat = get_feat_U(feat, dataset, start="t1", end="t2", flag=1, column_name="U_b1day_browse")
    feat = get_feat_U(feat, dataset, start="t1", end="t2", flag=2, column_name="U_b1day_collect")
    feat = get_feat_U(feat, dataset, start="t1", end="t2", flag=3, column_name="U_b1day_cart")
    feat = get_feat_U(feat, dataset, start="t1", end="t2", flag=4, column_name="U_b1day_buy")
    ##前2天
    feat = get_feat_U(feat, dataset, start="t3", end="t2", flag=1, column_name="U_b2day_browse")
    feat = get_feat_U(feat, dataset, start="t3", end="t2", flag=2, column_name="U_b2day_collect")
    feat = get_feat_U(feat, dataset, start="t3", end="t2", flag=3, column_name="U_b2day_cart")
    feat = get_feat_U(feat, dataset, start="t3", end="t2", flag=4, column_name="U_b2day_buy")
    ##前4天
    feat = get_feat_U(feat, dataset, start="t4", end="t2", flag=1, column_name="U_b4day_browse")
    feat = get_feat_U(feat, dataset, start="t4", end="t2", flag=2, column_name="U_b4day_collect")
    feat = get_feat_U(feat, dataset, start="t4", end="t2", flag=3, column_name="U_b4day_cart")
    feat = get_feat_U(feat, dataset, start="t4", end="t2", flag=4, column_name="U_b4day_buy")
    ##前7天
    feat = get_feat_U(feat, dataset, start="t5", end="t2", flag=1, column_name="U_b7day_browse")
    feat = get_feat_U(feat, dataset, start="t5", end="t2", flag=2, column_name="U_b7day_collect")
    feat = get_feat_U(feat, dataset, start="t5", end="t2", flag=3, column_name="U_b7day_cart")
    feat = get_feat_U(feat, dataset, start="t5", end="t2", flag=4, column_name="U_b7day_buy")
    ##28天
    feat = get_feat_U(feat, dataset, start="t6", end="t2", flag=1, column_name="U_b28day_browse")
    feat = get_feat_U(feat, dataset, start="t6", end="t2", flag=2, column_name="U_b28day_collect")
    feat = get_feat_U(feat, dataset, start="t6", end="t2", flag=3, column_name="U_b28day_cart")
    feat = get_feat_U(feat, dataset, start="t6", end="t2", flag=4, column_name="U_b28day_buy")


    ###商品特征
    ##前1天
    feat = get_feat_I(feat, dataset, start="t1", end="t2", flag=1, column_name="I_b1day_browse")
    feat = get_feat_I(feat, dataset, start="t1", end="t2", flag=2, column_name="I_b1day_collect")
    feat = get_feat_I(feat, dataset, start="t1", end="t2", flag=3, column_name="I_b1day_cart")
    feat = get_feat_I(feat, dataset, start="t1", end="t2", flag=4, column_name="I_b1day_buy")
    ##前2天
    feat = get_feat_I(feat, dataset, start="t3", end="t2", flag=1, column_name="I_b2day_browse")
    feat = get_feat_I(feat, dataset, start="t3", end="t2", flag=2, column_name="I_b2day_collect")
    feat = get_feat_I(feat, dataset, start="t3", end="t2", flag=3, column_name="I_b2day_cart")
    feat = get_feat_I(feat, dataset, start="t3", end="t2", flag=4, column_name="I_b2day_buy")
    ##前4天
    feat = get_feat_I(feat, dataset, start="t4", end="t2", flag=1, column_name="I_b4day_browse")
    feat = get_feat_I(feat, dataset, start="t4", end="t2", flag=2, column_name="I_b4day_collect")
    feat = get_feat_I(feat, dataset, start="t4", end="t2", flag=3, column_name="I_b4day_cart")
    feat = get_feat_I(feat, dataset, start="t4", end="t2", flag=4, column_name="I_b4day_buy")
    ##前7天
    feat = get_feat_I(feat, dataset, start="t5", end="t2", flag=1, column_name="I_b7day_browse")
    feat = get_feat_I(feat, dataset, start="t5", end="t2", flag=2, column_name="I_b7day_collect")
    feat = get_feat_I(feat, dataset, start="t5", end="t2", flag=3, column_name="I_b7day_cart")
    feat = get_feat_I(feat, dataset, start="t5", end="t2", flag=4, column_name="I_b7day_buy")
    ##28天
    feat = get_feat_I(feat, dataset, start="t6", end="t2", flag=1, column_name="I_b28day_browse")
    feat = get_feat_I(feat, dataset, start="t6", end="t2", flag=2, column_name="I_b28day_collect")
    feat = get_feat_I(feat, dataset, start="t6", end="t2", flag=3, column_name="I_b28day_cart")
    feat = get_feat_I(feat, dataset, start="t6", end="t2", flag=4, column_name="I_b28day_buy")


    ###类别特征
    ##前1天
    feat = get_feat_C(feat, dataset, start="t1", end="t2", flag=1, column_name="C_b1day_browse")
    feat = get_feat_C(feat, dataset, start="t1", end="t2", flag=2, column_name="C_b1day_collect")
    feat = get_feat_C(feat, dataset, start="t1", end="t2", flag=3, column_name="C_b1day_cart")
    feat = get_feat_C(feat, dataset, start="t1", end="t2", flag=4, column_name="C_b1day_buy")
    ##前2天
    feat = get_feat_C(feat, dataset, start="t3", end="t2", flag=1, column_name="C_b2day_browse")
    feat = get_feat_C(feat, dataset, start="t3", end="t2", flag=2, column_name="C_b2day_collect")
    feat = get_feat_C(feat, dataset, start="t3", end="t2", flag=3, column_name="C_b2day_cart")
    feat = get_feat_C(feat, dataset, start="t3", end="t2", flag=4, column_name="C_b2day_buy")
    ##前4天
    feat = get_feat_C(feat, dataset, start="t4", end="t2", flag=1, column_name="C_b4day_browse")
    feat = get_feat_C(feat, dataset, start="t4", end="t2", flag=2, column_name="C_b4day_collect")
    feat = get_feat_C(feat, dataset, start="t4", end="t2", flag=3, column_name="C_b4day_cart")
    feat = get_feat_C(feat, dataset, start="t4", end="t2", flag=4, column_name="C_b4day_buy")
    ##前7天
    feat = get_feat_C(feat, dataset, start="t5", end="t2", flag=1, column_name="C_b7day_browse")
    feat = get_feat_C(feat, dataset, start="t5", end="t2", flag=2, column_name="C_b7day_collect")
    feat = get_feat_C(feat, dataset, start="t5", end="t2", flag=3, column_name="C_b7day_cart")
    feat = get_feat_C(feat, dataset, start="t5", end="t2", flag=4, column_name="C_b7day_buy")
    ##28天
    feat = get_feat_C(feat, dataset, start="t6", end="t2", flag=1, column_name="C_b28day_browse")
    feat = get_feat_C(feat, dataset, start="t6", end="t2", flag=2, column_name="C_b28day_collect")
    feat = get_feat_C(feat, dataset, start="t6", end="t2", flag=3, column_name="C_b28day_cart")
    feat = get_feat_C(feat, dataset, start="t6", end="t2", flag=4, column_name="C_b28day_buy")


    ###UI特征
    ##前1天
    feat = get_feat_UI(feat, dataset, start="t1", end="t2", flag=1, column_name="UI_b1day_browse")
    feat = get_feat_UI(feat, dataset, start="t1", end="t2", flag=2, column_name="UI_b1day_collect")
    feat = get_feat_UI(feat, dataset, start="t1", end="t2", flag=3, column_name="UI_b1day_cart")
    feat = get_feat_UI(feat, dataset, start="t1", end="t2", flag=4, column_name="UI_b1day_buy")
    ##前2天
    feat = get_feat_UI(feat, dataset, start="t3", end="t2", flag=1, column_name="UI_b2day_browse")
    feat = get_feat_UI(feat, dataset, start="t3", end="t2", flag=2, column_name="UI_b2day_collect")
    feat = get_feat_UI(feat, dataset, start="t3", end="t2", flag=3, column_name="UI_b2day_cart")
    feat = get_feat_UI(feat, dataset, start="t3", end="t2", flag=4, column_name="UI_b2day_buy")
    ##前4天
    feat = get_feat_UI(feat, dataset, start="t4", end="t2", flag=1, column_name="UI_b4day_browse")
    feat = get_feat_UI(feat, dataset, start="t4", end="t2", flag=2, column_name="UI_b4day_collect")
    feat = get_feat_UI(feat, dataset, start="t4", end="t2", flag=3, column_name="UI_b4day_cart")
    feat = get_feat_UI(feat, dataset, start="t4", end="t2", flag=4, column_name="UI_b4day_buy")
    ##前7天
    feat = get_feat_UI(feat, dataset, start="t5", end="t2", flag=1, column_name="UI_b7day_browse")
    feat = get_feat_UI(feat, dataset, start="t5", end="t2", flag=2, column_name="UI_b7day_collect")
    feat = get_feat_UI(feat, dataset, start="t5", end="t2", flag=3, column_name="UI_b7day_cart")
    feat = get_feat_UI(feat, dataset, start="t5", end="t2", flag=4, column_name="UI_b7day_buy")
    ##28天
    feat = get_feat_UI(feat, dataset, start="t6", end="t2", flag=1, column_name="UI_b28day_browse")
    feat = get_feat_UI(feat, dataset, start="t6", end="t2", flag=2, column_name="UI_b28day_collect")
    feat = get_feat_UI(feat, dataset, start="t6", end="t2", flag=3, column_name="UI_b28day_cart")
    feat = get_feat_UI(feat, dataset, start="t6", end="t2", flag=4, column_name="UI_b28day_buy")


    ##UC特征
    ##前1天
    feat = get_feat_UC(feat, dataset, start="t1", end="t2", flag=1, column_name="UC_b1day_browse")
    feat = get_feat_UC(feat, dataset, start="t1", end="t2", flag=2, column_name="UC_b1day_collect")
    feat = get_feat_UC(feat, dataset, start="t1", end="t2", flag=3, column_name="UC_b1day_cart")
    feat = get_feat_UC(feat, dataset, start="t1", end="t2", flag=4, column_name="UC_b1day_buy")
    ##前2天
    feat = get_feat_UC(feat, dataset, start="t3", end="t2", flag=1, column_name="UC_b2day_browse")
    feat = get_feat_UC(feat, dataset, start="t3", end="t2", flag=2, column_name="UC_b2day_collect")
    feat = get_feat_UC(feat, dataset, start="t3", end="t2", flag=3, column_name="UC_b2day_cart")
    feat = get_feat_UC(feat, dataset, start="t3", end="t2", flag=4, column_name="UC_b2day_buy")
    ##前4天
    feat = get_feat_UC(feat, dataset, start="t4", end="t2", flag=1, column_name="UC_b4day_browse")
    feat = get_feat_UC(feat, dataset, start="t4", end="t2", flag=2, column_name="UC_b4day_collect")
    feat = get_feat_UC(feat, dataset, start="t4", end="t2", flag=3, column_name="UC_b4day_cart")
    feat = get_feat_UC(feat, dataset, start="t4", end="t2", flag=4, column_name="UC_b4day_buy")
    ##前7天
    feat = get_feat_UC(feat, dataset, start="t5", end="t2", flag=1, column_name="UC_b7day_browse")
    feat = get_feat_UC(feat, dataset, start="t5", end="t2", flag=2, column_name="UC_b7day_collect")
    feat = get_feat_UC(feat, dataset, start="t5", end="t2", flag=3, column_name="UC_b7day_cart")
    feat = get_feat_UC(feat, dataset, start="t5", end="t2", flag=4, column_name="UC_b7day_buy")
    ##前28天
    feat = get_feat_UC(feat, dataset, start="t6", end="t2", flag=1, column_name="UC_b28day_browse")
    feat = get_feat_UC(feat, dataset, start="t6", end="t2", flag=2, column_name="UC_b28day_collect")
    feat = get_feat_UC(feat, dataset, start="t6", end="t2", flag=3, column_name="UC_b28day_cart")
    feat = get_feat_UC(feat, dataset, start="t6", end="t2", flag=4, column_name="UC_b28day_buy")


    ##前1天
    feat = get_feat_or_not(feat, dataset, start="t1", end="t2", flag=1, column_name="UI_b1day_browse_or_not")
    feat = get_feat_or_not(feat, dataset, start="t1", end="t2", flag=2, column_name="UI_b1day_collect_or_not")
    feat = get_feat_or_not(feat, dataset, start="t1", end="t2", flag=3, column_name="UI_b1day_cart_or_not")
    feat = get_feat_or_not(feat, dataset, start="t1", end="t2", flag=4, column_name="UI_b1day_buy_or_not")
    ##前2天
    feat = get_feat_or_not(feat, dataset, start="t3", end="t1", flag=1, column_name="UI_b2day_browse_or_not")
    feat = get_feat_or_not(feat, dataset, start="t3", end="t1", flag=2, column_name="UI_b2day_collect_or_not")
    feat = get_feat_or_not(feat, dataset, start="t3", end="t1", flag=3, column_name="UI_b2day_cart_or_not")
    feat = get_feat_or_not(feat, dataset, start="t3", end="t1", flag=4, column_name="UI_b2day_buy_or_not")
    ##前4天
    feat = get_feat_or_not(feat, dataset, start="t4", end="t3", flag=1, column_name="UI_b4day_browse_or_not")
    feat = get_feat_or_not(feat, dataset, start="t4", end="t3", flag=2, column_name="UI_b4day_collect_or_not")
    feat = get_feat_or_not(feat, dataset, start="t4", end="t3", flag=3, column_name="UI_b4day_cart_or_not")
    feat = get_feat_or_not(feat, dataset, start="t4", end="t3", flag=4, column_name="UI_b4day_buy_or_not")
    ##前7天
    feat = get_feat_or_not(feat, dataset, start="t5", end="t4", flag=1, column_name="UI_b7day_browse_or_not")
    feat = get_feat_or_not(feat, dataset, start="t5", end="t4", flag=2, column_name="UI_b7day_collect_or_not")
    feat = get_feat_or_not(feat, dataset, start="t5", end="t4", flag=3, column_name="UI_b7day_cart_or_not")
    feat = get_feat_or_not(feat, dataset, start="t5", end="t4", flag=4, column_name="UI_b7day_buy_or_not")


    ##前1天
    feat = get_feat_other_i(feat, dataset, start="t1", end="t2", flag=1, column_name="UC_b1day_browse_or_not")
    feat = get_feat_other_i(feat, dataset, start="t1", end="t2", flag=2, column_name="UC_b1day_collect_or_not")
    feat = get_feat_other_i(feat, dataset, start="t1", end="t2", flag=3, column_name="UC_b1day_cart_or_not")
    feat = get_feat_other_i(feat, dataset, start="t1", end="t2", flag=4, column_name="UC_b1day_buy_or_not")
    ##前2天
    feat = get_feat_other_i(feat, dataset, start="t3", end="t1", flag=1, column_name="UC_b2day_browse_or_not")
    feat = get_feat_other_i(feat, dataset, start="t3", end="t1", flag=2, column_name="UC_b2day_collect_or_not")
    feat = get_feat_other_i(feat, dataset, start="t3", end="t1", flag=3, column_name="UC_b2day_cart_or_not")
    feat = get_feat_other_i(feat, dataset, start="t3", end="t1", flag=4, column_name="UC_b2day_buy_or_not")
    ##前4天
    feat = get_feat_other_i(feat, dataset, start="t4", end="t3", flag=1, column_name="UC_b4day_browse_or_not")
    feat = get_feat_other_i(feat, dataset, start="t4", end="t3", flag=2, column_name="UC_b4day_collect_or_not")
    feat = get_feat_other_i(feat, dataset, start="t4", end="t3", flag=3, column_name="UC_b4day_cart_or_not")
    feat = get_feat_other_i(feat, dataset, start="t4", end="t3", flag=4, column_name="UC_b4day_buy_or_not")
    ##前7天
    feat = get_feat_other_i(feat, dataset, start="t5", end="t4", flag=1, column_name="UC_b7day_browse_or_not")
    feat = get_feat_other_i(feat, dataset, start="t5", end="t4", flag=2, column_name="UC_b7day_collect_or_not")
    feat = get_feat_other_i(feat, dataset, start="t5", end="t4", flag=3, column_name="UC_b7day_cart_or_not")
    feat = get_feat_other_i(feat, dataset, start="t5", end="t4", flag=4, column_name="UC_b7day_buy_or_not")


    ###交叉特征
    ##UI / UC
    feat["b1day_UI_UC_browse_rate"] = feat["UI_b1day_browse"] / (feat["UC_b1day_browse"]+1e-16)
    feat["b2day_UI_UC_browse_rate"] = feat["UI_b2day_browse"] / (feat["UC_b2day_browse"]+1e-16)
    feat["b4day_UI_UC_browse_rate"] = feat["UI_b4day_browse"] / (feat["UC_b4day_browse"]+1e-16)
    feat["b7day_UI_UC_browse_rate"] = feat["UI_b7day_browse"] / (feat["UC_b7day_browse"]+1e-16)
    feat["b28day_UI_UC_browse_rate"] = feat["UI_b28day_browse"] / (feat["UC_b28day_browse"]+1e-16)

    feat["b1day_UI_UC_collect_rate"] = feat["UI_b1day_collect"] / (feat["UC_b1day_collect"]+1e-16)
    feat["b2day_UI_UC_collect_rate"] = feat["UI_b2day_collect"] / (feat["UC_b2day_collect"]+1e-16)
    feat["b4day_UI_UC_collect_rate"] = feat["UI_b4day_collect"] / (feat["UC_b4day_collect"]+1e-16)
    feat["b7day_UI_UC_collect_rate"] = feat["UI_b7day_collect"] / (feat["UC_b7day_collect"]+1e-16)
    feat["b28day_UI_UC_collect_rate"] = feat["UI_b28day_collect"] / (feat["UC_b28day_collect"]+1e-16)

    feat["b1day_UI_UC_cart_rate"] = feat["UI_b1day_cart"] / (feat["UC_b1day_cart"]+1e-16)
    feat["b2day_UI_UC_cart_rate"] = feat["UI_b2day_cart"] / (feat["UC_b2day_cart"]+1e-16)
    feat["b4day_UI_UC_cart_rate"] = feat["UI_b4day_cart"] / (feat["UC_b4day_cart"]+1e-16)
    feat["b7day_UI_UC_cart_rate"] = feat["UI_b7day_cart"] / (feat["UC_b7day_cart"]+1e-16)
    feat["b28day_UI_UC_cart_rate"] = feat["UI_b28day_cart"] / (feat["UC_b28day_cart"]+1e-16)

    feat["b1day_UI_UC_buy_rate"] = feat["UI_b1day_buy"] / (feat["UC_b1day_buy"]+1e-16)
    feat["b2day_UI_UC_buy_rate"] = feat["UI_b2day_buy"] / (feat["UC_b2day_buy"]+1e-16)
    feat["b4day_UI_UC_buy_rate"] = feat["UI_b4day_buy"] / (feat["UC_b4day_buy"]+1e-16)
    feat["b7day_UI_UC_buy_rate"] = feat["UI_b7day_buy"] / (feat["UC_b7day_buy"]+1e-16)
    feat["b28day_UI_UC_buy_rate"] = feat["UI_b28day_buy"] / (feat["UC_b28day_buy"]+1e-16)


    ### UI / I
    feat["b1day_UI_I_browse_rate"] = feat["UI_b1day_browse"] / (feat["I_b1day_browse"]+1e-16)
    feat["b2day_UI_I_browse_rate"] = feat["UI_b2day_browse"] / (feat["I_b2day_browse"]+1e-16)
    feat["b4day_UI_I_browse_rate"] = feat["UI_b4day_browse"] / (feat["I_b4day_browse"]+1e-16)
    feat["b7day_UI_I_browse_rate"] = feat["UI_b7day_browse"] / (feat["I_b7day_browse"]+1e-16)
    feat["b28day_UI_I_browse_rate"] = feat["UI_b28day_browse"] / (feat["I_b28day_browse"]+1e-16)

    feat["b1day_UI_I_collect_rate"] = feat["UI_b1day_collect"] / (feat["I_b1day_collect"]+1e-16)
    feat["b2day_UI_I_collect_rate"] = feat["UI_b2day_collect"] / (feat["I_b2day_collect"]+1e-16)
    feat["b4day_UI_I_collect_rate"] = feat["UI_b4day_collect"] / (feat["I_b4day_collect"]+1e-16)
    feat["b7day_UI_I_collect_rate"] = feat["UI_b7day_collect"] / (feat["I_b7day_collect"]+1e-16)
    feat["b28day_UI_I_collect_rate"] = feat["UI_b28day_collect"] / (feat["I_b28day_collect"]+1e-16)

    feat["b1day_UI_I_cart_rate"] = feat["UI_b1day_cart"] / (feat["I_b1day_cart"]+1e-16)
    feat["b2day_UI_I_cart_rate"] = feat["UI_b2day_cart"] / (feat["I_b2day_cart"]+1e-16)
    feat["b4day_UI_I_cart_rate"] = feat["UI_b4day_cart"] / (feat["I_b4day_cart"]+1e-16)
    feat["b7day_UI_I_cart_rate"] = feat["UI_b7day_cart"] / (feat["I_b7day_cart"]+1e-16)
    feat["b28day_UI_I_cart_rate"] = feat["UI_b28day_cart"] / (feat["I_b28day_cart"]+1e-16)

    feat["b1day_UI_I_buy_rate"] = feat["UI_b1day_buy"] / (feat["I_b1day_buy"]+1e-16)
    feat["b2day_UI_I_buy_rate"] = feat["UI_b2day_buy"] / (feat["I_b2day_buy"]+1e-16)
    feat["b4day_UI_I_buy_rate"] = feat["UI_b4day_buy"] / (feat["I_b4day_buy"]+1e-16)
    feat["b7day_UI_I_buy_rate"] = feat["UI_b7day_buy"] / (feat["I_b7day_buy"]+1e-16)
    feat["b28day_UI_I_buy_rate"] = feat["UI_b28day_buy"] / (feat["I_b28day_buy"]+1e-16)


    ### UI /U
    feat["b1day_UI_U_browse_rate"] = feat["UI_b1day_browse"] / (feat["U_b1day_browse"]+1e-16)
    feat["b2day_UI_U_browse_rate"] = feat["UI_b2day_browse"] / (feat["U_b2day_browse"]+1e-16)
    feat["b4day_UI_U_browse_rate"] = feat["UI_b4day_browse"] / (feat["U_b4day_browse"]+1e-16)
    feat["b7day_UI_U_browse_rate"] = feat["UI_b7day_browse"] / (feat["U_b7day_browse"]+1e-16)
    feat["b28day_UI_U_browse_rate"] = feat["UI_b28day_browse"] / (feat["U_b28day_browse"]+1e-16)

    feat["b1day_UI_U_collect_rate"] = feat["UI_b1day_collect"] / (feat["U_b1day_collect"]+1e-16)
    feat["b2day_UI_U_collect_rate"] = feat["UI_b2day_collect"] / (feat["U_b2day_collect"]+1e-16)
    feat["b4day_UI_U_collect_rate"] = feat["UI_b4day_collect"] / (feat["U_b4day_collect"]+1e-16)
    feat["b7day_UI_U_collect_rate"] = feat["UI_b7day_collect"] / (feat["U_b7day_collect"]+1e-16)
    feat["b28day_UI_U_collect_rate"] = feat["UI_b28day_collect"] / (feat["U_b28day_collect"]+1e-16)

    feat["b1day_UI_U_cart_rate"] = feat["UI_b1day_cart"] / (feat["U_b1day_cart"]+1e-16)
    feat["b2day_UI_U_cart_rate"] = feat["UI_b2day_cart"] / (feat["U_b2day_cart"]+1e-16)
    feat["b4day_UI_U_cart_rate"] = feat["UI_b4day_cart"] / (feat["U_b4day_cart"]+1e-16)
    feat["b7day_UI_U_cart_rate"] = feat["UI_b7day_cart"] / (feat["U_b7day_cart"]+1e-16)
    feat["b28day_UI_U_cart_rate"] = feat["UI_b28day_cart"] / (feat["U_b28day_cart"]+1e-16)

    feat["b1day_UI_U_buy_rate"] = feat["UI_b1day_buy"] / (feat["U_b1day_buy"]+1e-16)
    feat["b2day_UI_U_buy_rate"] = feat["UI_b2day_buy"] / (feat["U_b2day_buy"]+1e-16)
    feat["b4day_UI_U_buy_rate"] = feat["UI_b4day_buy"] / (feat["U_b4day_buy"]+1e-16)
    feat["b7day_UI_U_buy_rate"] = feat["UI_b7day_buy"] / (feat["U_b7day_buy"]+1e-16)
    feat["b28day_UI_U_buy_rate"] = feat["UI_b28day_buy"] / (feat["U_b28day_buy"]+1e-16)


    ##UC /U
    feat["b1day_UC_U_browse_rate"] = feat["UC_b1day_browse"] / (feat["U_b1day_browse"]+1e-16)
    feat["b2day_UC_U_browse_rate"] = feat["UC_b2day_browse"] / (feat["U_b2day_browse"]+1e-16)
    feat["b4day_UC_U_browse_rate"] = feat["UC_b4day_browse"] / (feat["U_b4day_browse"]+1e-16)
    feat["b7day_UC_U_browse_rate"] = feat["UC_b7day_browse"] / (feat["U_b7day_browse"]+1e-16)
    feat["b28day_UC_U_browse_rate"] = feat["UC_b28day_browse"] / (feat["U_b28day_browse"]+1e-16)

    feat["b1day_UC_U_collect_rate"] = feat["UC_b1day_collect"] / (feat["U_b1day_collect"]+1e-16)
    feat["b2day_UC_U_collect_rate"] = feat["UC_b2day_collect"] / (feat["U_b2day_collect"]+1e-16)
    feat["b4day_UC_U_collect_rate"] = feat["UC_b4day_collect"] / (feat["U_b4day_collect"]+1e-16)
    feat["b7day_UC_U_collect_rate"] = feat["UC_b7day_collect"] / (feat["U_b7day_collect"]+1e-16)
    feat["b28day_UC_U_collect_rate"] = feat["UC_b28day_collect"] / (feat["U_b28day_collect"]+1e-16)

    feat["b1day_UC_U_cart_rate"] = feat["UC_b1day_cart"] / (feat["U_b1day_cart"]+1e-16)
    feat["b2day_UC_U_cart_rate"] = feat["UC_b2day_cart"] / (feat["U_b2day_cart"]+1e-16)
    feat["b4day_UC_U_cart_rate"] = feat["UC_b4day_cart"] / (feat["U_b4day_cart"]+1e-16)
    feat["b7day_UC_U_cart_rate"] = feat["UC_b7day_cart"] / (feat["U_b7day_cart"]+1e-16)
    feat["b28day_UC_U_cart_rate"] = feat["UC_b28day_cart"] / (feat["U_b28day_cart"]+1e-16)

    feat["b1day_UC_U_buy_rate"] = feat["UC_b1day_buy"] / (feat["U_b1day_buy"]+1e-16)
    feat["b2day_UC_U_buy_rate"] = feat["UC_b2day_buy"] / (feat["U_b2day_buy"]+1e-16)
    feat["b4day_UC_U_buy_rate"] = feat["UC_b4day_buy"] / (feat["U_b4day_buy"]+1e-16)
    feat["b7day_UC_U_buy_rate"] = feat["UC_b7day_buy"] / (feat["U_b7day_buy"]+1e-16)
    feat["b28day_UC_U_buy_rate"] = feat["UC_b28day_buy"] / (feat["U_b28day_buy"]+1e-16)


    ##UC / C
    feat["b1day_UC_C_browse_rate"] = feat["UC_b1day_browse"] / (feat["C_b1day_browse"]+1e-16)
    feat["b2day_UC_C_browse_rate"] = feat["UC_b2day_browse"] / (feat["C_b2day_browse"]+1e-16)
    feat["b4day_UC_C_browse_rate"] = feat["UC_b4day_browse"] / (feat["C_b4day_browse"]+1e-16)
    feat["b7day_UC_C_browse_rate"] = feat["UC_b7day_browse"] / (feat["C_b7day_browse"]+1e-16)
    feat["b28day_UC_C_browse_rate"] = feat["UC_b28day_browse"] / (feat["C_b28day_browse"]+1e-16)

    feat["b1day_UC_C_collect_rate"] = feat["UC_b1day_collect"] / (feat["C_b1day_collect"]+1e-16)
    feat["b2day_UC_C_collect_rate"] = feat["UC_b2day_collect"] / (feat["C_b2day_collect"]+1e-16)
    feat["b4day_UC_C_collect_rate"] = feat["UC_b4day_collect"] / (feat["C_b4day_collect"]+1e-16)
    feat["b7day_UC_C_collect_rate"] = feat["UC_b7day_collect"] / (feat["C_b7day_collect"]+1e-16)
    feat["b28day_UC_C_collect_rate"] = feat["UC_b28day_collect"] / (feat["C_b28day_collect"]+1e-16)

    feat["b1day_UC_C_cart_rate"] = feat["UC_b1day_cart"] / (feat["C_b1day_cart"]+1e-16)
    feat["b2day_UC_C_cart_rate"] = feat["UC_b2day_cart"] / (feat["C_b2day_cart"]+1e-16)
    feat["b4day_UC_C_cart_rate"] = feat["UC_b4day_cart"] / (feat["C_b4day_cart"]+1e-16)
    feat["b7day_UC_C_cart_rate"] = feat["UC_b7day_cart"] / (feat["C_b7day_cart"]+1e-16)
    feat["b28day_UC_C_cart_rate"] = feat["UC_b28day_cart"] / (feat["C_b28day_cart"]+1e-16)

    feat["b1day_UC_C_buy_rate"] = feat["UC_b1day_buy"] / (feat["C_b1day_buy"]+1e-16)
    feat["b2day_UC_C_buy_rate"] = feat["UC_b2day_buy"] / (feat["C_b2day_buy"]+1e-16)
    feat["b4day_UC_C_buy_rate"] = feat["UC_b4day_buy"] / (feat["C_b4day_buy"]+1e-16)
    feat["b7day_UC_C_buy_rate"] = feat["UC_b7day_buy"] / (feat["C_b7day_buy"]+1e-16)
    feat["b28day_UC_C_buy_rate"] = feat["UC_b28day_buy"] / (feat["C_b28day_buy"]+1e-16)

    feat["b1day_I_C_browse_rate"] = feat["I_b1day_browse"] / (feat["C_b1day_browse"]+1e-16)
    feat["b2day_I_C_browse_rate"] = feat["I_b2day_browse"] / (feat["C_b2day_browse"]+1e-16)
    feat["b4day_I_C_browse_rate"] = feat["I_b4day_browse"] / (feat["C_b4day_browse"]+1e-16)
    feat["b7day_I_C_browse_rate"] = feat["I_b7day_browse"] / (feat["C_b7day_browse"]+1e-16)
    feat["b28day_I_C_browse_rate"] = feat["I_b28day_browse"] / (feat["C_b28day_browse"]+1e-16)

    feat["b1day_I_C_collect_rate"] = feat["I_b1day_collect"] / (feat["C_b1day_collect"]+1e-16)
    feat["b2day_I_C_collect_rate"] = feat["I_b2day_collect"] / (feat["C_b2day_collect"]+1e-16)
    feat["b4day_I_C_collect_rate"] = feat["I_b4day_collect"] / (feat["C_b4day_collect"]+1e-16)
    feat["b7day_I_C_collect_rate"] = feat["I_b7day_collect"] / (feat["C_b7day_collect"]+1e-16)
    feat["b28day_I_C_collect_rate"] = feat["I_b28day_collect"] / (feat["C_b28day_collect"]+1e-16)

    feat["b1day_I_C_cart_rate"] = feat["I_b1day_cart"] / (feat["C_b1day_cart"]+1e-16)
    feat["b2day_I_C_cart_rate"] = feat["I_b2day_cart"] / (feat["C_b2day_cart"]+1e-16)
    feat["b4day_I_C_cart_rate"] = feat["I_b4day_cart"] / (feat["C_b4day_cart"]+1e-16)
    feat["b7day_I_C_cart_rate"] = feat["I_b7day_cart"] / (feat["C_b7day_cart"]+1e-16)
    feat["b28day_I_C_cart_rate"] = feat["I_b28day_cart"] / (feat["C_b28day_cart"]+1e-16)

    feat["b1day_I_C_buy_rate"] = feat["I_b1day_buy"] / (feat["C_b1day_buy"]+1e-16)
    feat["b2day_I_C_buy_rate"] = feat["I_b2day_buy"] / (feat["C_b2day_buy"]+1e-16)
    feat["b4day_I_C_buy_rate"] = feat["I_b4day_buy"] / (feat["C_b4day_buy"]+1e-16)
    feat["b7day_I_C_buy_rate"] = feat["I_b7day_buy"] / (feat["C_b7day_buy"]+1e-16)
    feat["b28day_I_C_buy_rate"] = feat["I_b28day_buy"] / (feat["C_b28day_buy"]+1e-16)

    return feat
