from pycaret.classification import setup, tune_model
import os
import pandas as pd
import numpy as np
from collections import defaultdict
# import xgboost as xgb
# import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from embedding import prepare_data
from alive_progress import alive_bar

tuned_model_list_datas = defaultdict()
i = 0

emb_types = ['ast_', 'combined_', 'code_']

emb_type = "code_"
data = pd.read_pickle('../../data/prepared/run2/train.pkl')
print(f'--> {emb_type}')
# filtered_data = data.loc[:, data.columns.str.startswith(emb_type)]
# filtered_data['actual label'] = data['actual label']

filtered_data['code_embeddings'] = filtered_data['code_embeddings'].apply(lambda x: x.flatten())

filtered_data = prepare_data(data, 'code_embeddings', 'actual label')

print("flattened data")
# print(filtered_data.shape)
reg1 = setup(data = filtered_data, target = 'actual label', index=False, session_id=42)
# Set models to tune
params = [["learning_rate", [1.0, 0.5, 0.1]], [ "n_estimators", [25, 50, 100, 150, 200, 400]], ["loss", ['log_loss', 'exponential']], ["criterion", ['friedman_mse', 'squared_error']]]

processed_params = [{}]
for category in params:
    print(category)
    param = category[0]
    options = category[1]
    new_processed_params = []
    for processed_param in processed_params:
        for option in options:
            new_processed_param = processed_param.copy()
            new_processed_param[param] = option
            new_processed_params.append(new_processed_param)
    processed_params = new_processed_params

print(processed_params)
nn = True
model_list = []
if nn:
    model_list = [MLPClassifier()]
else:
    model = GradientBoostingClassifier(**processed_param)
    model_list.append(model)

print(len(model_list))
tuned_model_list = []

with alive_bar(len(model_list)) as bar:
    for model in model_list:
        tuned_model = tune_model(model)
        tuned_model_list.append(tuned_model)
        bar()

tuned_model_list_datas[emb_type] = tuned_model_list
i += 1



# for folder_name in os.listdir(''):
#     print(folder_name)
#     file_name = [x for x in os.listdir('') if 'train' in x][0]
#     data = pd.read_csv('', index_col=0)
#     for emb_type in emb_types:
#         print(f'--> {emb_type}')

#         filtered_data = data.loc[:, data.columns.str.startswith(emb_type)]
#         filtered_data['actual label'] = data['actual label']
#         print(filtered_data.shape)
#         reg1 = setup(data = filtered_data, target = 'actual label', index=False, session_id=42)
#         # Set models to tune
#         model_list = [GradientBoostingClassifier()]
#         tuned_model_list = []

#         for model in model_list:
#             tuned_model = tune_model(model)
#             tuned_model_list.append(tuned_model)
#             print(tuned_model, '\n')
        
#         tuned_model_list_datas[folder_name+emb_type] = tuned_model_list
#         i += 1


with open('../../data/models-nn.file', 'wb') as f:
    pickle.dump(tuned_model_list_datas, f)