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
from utils.embedding import prepare_data
from alive_progress import alive_bar
from utils.file_retrieval import DataFileDirectory
from utils.directories import models_dir, prepared_dir

def get_processed_params():
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
    return processed_params

tuned_model_list_datas = defaultdict()
i = 0

# emb_types = ['ast_', 'combined_', 'code_']




if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
    model_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + models_dir
    data = DataFileDirectory(file_path, '.emb.pkl')

    train_file_name = data.get_file("Select file to use as training")

    emb_type = "code_"

    data = pd.read_pickle(train_file_name)
    print(f'--> {emb_type}')

    filtered_data = prepare_data(data, 'code_embeddings', 'actual label')

    reg1 = setup(data = filtered_data, target = 'actual label', index=False, session_id=42)

    nn = False
    model_list = []
    if nn:
        model_list = [MLPClassifier()]
    else:
        model = GradientBoostingClassifier()
        model_list.append(model)

    print(len(model_list))
    tuned_model_list = []

    with alive_bar(len(model_list)) as bar:
        for model in model_list:
            tuned_model = tune_model(model, verbose=False)
            tuned_model_list.append(tuned_model)
            bar()

    tuned_model_list_datas[emb_type] = tuned_model_list
    i += 1

    model_files = [x[:-5] for x in os.listdir(model_dir) if x.endswith(".file")]
    print("Found existing models: ")
    for idx, file in enumerate(model_files):
        print(f"{idx+1}. {file}")
    train_file = input("File number to load as training: ")
    if(train_file.isnumeric() and int(train_file) <= len(model_files)):
        data_output_file = model_dir + model_files[int(train_file)-1] + ".file"
    else:
        data_output_file = model_dir + train_file + ".file"

    with open(data_output_file, 'wb') as f:
        pickle.dump(tuned_model_list_datas, f)