# coding: utf-8

import pandas as pd
from extract_feat import extract_feat

dataset = pd.read_csv("extract_dataset/extract_feat_dataset_18.csv")
feat = dataset[dataset.time >= "t5"][["user_id", "item_id", "item_category"]]
feat = feat.drop_duplicates()
feat = extract_feat(feat, dataset, '2014-12-18 00',
                                    '2014-12-19 00',
                                    '2014-12-17 00',
                                    '2014-12-15 00',
                                    '2014-12-12 00',
                                    '2014-11-18 00')

feat.to_csv("feature/feature4.csv",index=False)





