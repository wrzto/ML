# coding: utf-8

import pandas as pd
from extract_feat import extract_feat

dataset = pd.read_csv("extract_dataset/extract_feat_dataset_17.csv")
feat = dataset[dataset.time >= "t5"][["user_id", "item_id", "item_category", "label"]]
feat = feat.drop_duplicates()
feat = extract_feat(feat, dataset, '2014-12-17 00',
                                    '2014-12-18 00',
                                    '2014-12-16 00',
                                    '2014-12-14 00',
                                    '2014-12-11 00',
                                    '2014-11-18 00')

feat.to_csv("feature/feature3.csv",index=False)





