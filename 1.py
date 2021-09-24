from tqdm import tqdm

tqdm.pandas(desc='pandas bar')
import gc
import numpy as np
from evaluation import uAUC
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
from try2 import fmin, tpe, hp, partial
from sklearn.metrics import mean_squared_error


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


path = './wechat_algo_data1/'
train = pd.read_csv(path + 'user_action.csv')
test = pd.read_csv(path + 'test_a.csv')
feed_info = pd.read_csv(path + 'feed_info.csv')
res = test[['userid', 'feedid']]
feed_info = feed_info.fillna('')

feed_one_hot_feature = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# feed_vector_feature = [description'', 'ocr', 'asr','machine_keyword_list', 'manual_keyword_list', 'manual_tag_list', 'machine_tag_list','description_char', 'ocr_char', 'asr_char']
feed_vector_feature = ['description']

# 替换分隔符
# for col in feed_vector_feature:
#     feed_info[col] = feed_info[col].apply(lambda x:str(x).replace(';',' '))

train = pd.merge(train, feed_info, on=['feedid'], how='left', copy=False)
test = pd.merge(test, feed_info, on=['feedid'], how='left', copy=False)

test['date_'] = 15

data = pd.concat([train, test], axis=0, copy=False)

label_columns = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', ]
drop_columns = ['play', 'stay', 'date_']
action_one_hot_feature = ['userid', 'device']

one_hot_feature = feed_one_hot_feature + action_one_hot_feature

for feature in one_hot_feature:
    data[feature] = LabelEncoder().fit_transform(data[feature].apply(str))

train_shape = data[data['date_'] < 14].shape[0]
valid_shape = data[data['date_'] == 14].shape[0]

train = data[data['date_'] < 15]
train_y = data[data['date_'] < 15][label_columns]

x_train_shape = train[train['date_'] != 14].shape[0]

test = data[data['date_'] == 15]

enc = OneHotEncoder()
for index, feature in enumerate(one_hot_feature):
    print(feature)
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a = enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    if index == 0:
        train_x = train_a
        test_x = test_a
        print(train_x)
    else:
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

train = train_x.tocsr()
X_test = test_x.tocsr()

X_train = train[:train_shape]
X_valid = train[train_shape:]

print(X_train.shape, X_valid.shape, X_test.shape)

y_train = train_y[:train_shape]
y_valid = train_y[train_shape:]
print(y_train.shape, y_valid.shape)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 6,
    'num_leaves': 70,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 45,
    'n_jobs': -1,
    'min_child_samples': 21,
    'min_child_weight': 0.001,
    'reg_alpha' : 0.001,
    'reg_lambda' : 8,
    'cat_smooth' : 0,
    'force_col_wise': True,
    # 'two_round': True
}

# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 15),
         "num_trees": hp.randint("num_trees", 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "bagging_fraction": hp.randint("bagging_fraction", 5),
         "num_leaves": hp.randint("num_leaves", 6),
         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['num_trees'] = argsDict['num_trees'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["bagging_fraction"] = argsDict["bagging_fraction"] * 0.1 + 0.5
    argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict

auc_sum = 0
bst_item = []
ul = data[data['date_'] == 14]['userid'].tolist()
weight_label = [4, 3, 2, 1]
for index, label in enumerate(['read_comment', 'like', 'click_avatar', 'forward']):
    print(label, index)
    dtrain = lgb.Dataset(X_train, label=y_train[label].values)
    dval = lgb.Dataset(X_valid, label=y_valid[label].values)
    lgb_model = LGBMRegressor(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    X_valid_pred = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    bst_item.append(lgb_model.best_iteration)
    v = uAUC(y_valid[label].tolist(), X_valid_pred.tolist(), ul)
    print(label, v)
    auc_sum = auc_sum + weight_label[index] * v / np.sum(weight_label)

print(auc_sum)

submit = res
# for index, label in enumerate(['read_comment', 'like', 'click_avatar', 'forward']):
#     dtrain = lgb.Dataset(train, label=train_y[label])
#     lgb_model = LGBMRegressor(
#         params,
#         dtrain,
#         num_boost_round=bst_item[index],
#         valid_sets=[dtrain],
#         early_stopping_rounds=50,
#         verbose_eval=50,
#     )
#
#     X_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
#     submit[label] = X_test_pred
# submit.to_csv('./baseline_{}.csv'.format(str(auc_sum).split('.')[1]), index=False)
