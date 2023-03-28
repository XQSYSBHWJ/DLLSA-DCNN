# @Time : 2023/1/5 7:54:23
# @Author : 海晏河清
# @File : 连接数据和pka.py  
# The environment here is TensorFlow (tf2)
# conda 4.12.0
# Python 3.7.11
# torch 1.7.1
# tensorflow 2.1.0
#  <(￣︶￣)↗[GO!]     
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import pandas as pd
import numpy as np
from argparse import RawDescriptionHelpFormatter
"""--------------------------------------------------数据集和pka结合------------------------------------------"""

if __name__ == "__main__":
    print("Start Now ...")
    d = """结合数据集和pka值"""
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_features", type=str, default="Onion1_D2020all_all_train.csv",
                        help="Input. Specify the features.")
    parser.add_argument("-inp_true", type=str, default="Pka_all_dataset_true.csv",
                        help="Input. Specify all complexes pKa.")
    parser.add_argument("-out", type=str, default="Onion1_D2020_all_pka_train.csv",
                        help="Output. Specify the target file name.")
    args = parser.parse_args()
    # 特征数值加载
    features = pd.read_csv(args.inp_features, index_col=0)
    index = features.index.tolist()
    values = features.values
    # 真实pka值加载
    true_pka = pd.read_csv(args.inp_true, index_col=0)
    true_index = true_pka.index.tolist()
    true_values = true_pka.values

    dict_ = {}
    for k, v in zip(index, values):
        dict_[k] = v.tolist()

    true_dict = {}
    for k, v in zip(true_index, true_values):
        true_dict[k] = v.tolist()

    new_values = []
    for ii in index:
        for jj in true_index:
            if str(jj) in str(ii):
                new_values.append(dict_[ii] + true_dict[jj])
                print(ii,'is ok')

    print(len(new_values))
    print(len(new_values[0]))

    columns = features.columns.tolist() + ['pKa']

df = pd.DataFrame(np.array(new_values), index=index, columns=columns)
df.to_csv(args.out)