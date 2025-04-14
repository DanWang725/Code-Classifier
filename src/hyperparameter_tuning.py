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
from utils.embedding import prepare_data, EncoderMap, ENCODERS
from alive_progress import alive_bar
from utils.file_retrieval import DataFileDirectory
from utils.directories import models_dir, prepared_dir
from utils.file_utils import get_emb_stats, EMBEDDING_EXTENSION

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
    data_loader = DataFileDirectory(file_path, EMBEDDING_EXTENSION, get_emb_stats, {'end': '.train'})

    train_file_path = data_loader.get_file("Select file to use as training")
    file = ''
    while(file is not None):
        file = data_loader.get_file('Select file to use as training')

    print("What is the encoding??")
    for index, label in enumerate(EncoderMap.values()):
        print(f"{index}. {label}")

    embedder = EncoderMap[ENCODERS[int(input())]]

    for index, train_file in enumerate(data_loader.get_chosen_files(prefix_path=True, extension=True)):
        print(f"Loading {train_file}")
        data = pd.read_pickle(train_file)

        filtered_data = prepare_data(data, 'code_embeddings', 'actual label')

        reg1 = setup(data = filtered_data, target = 'actual label', index=False, session_id=42)

        models = {'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(filtered_data.shape[0] - 1,)), 'GradientBoosted': GradientBoostingClassifier(ccp_alpha=0.001)}
        # models = {'NeuralNetwork': MLPClassifier(), 'GradientBoosted': GradientBoostingClassifier()}

        data_package = {'embedding': embedder, 'train_file': data_loader.get_chosen_files(extension=True)[index], 'models': {}}
        print(data_package)

        with alive_bar(len(models.items())) as bar:
            for model_name, model in models.items():
                tuned_model = tune_model(model, verbose=False)
                data_package['models'][model_name] = tuned_model
                bar()

        data_output_file = model_dir + data_loader.get_chosen_files()[index] + ".model"

        # model_files = [x[:-6] for x in os.listdir(model_dir) if x.endswith(".model")]

    # print("Found existing models: ")
    # for idx, file in enumerate(model_files):
    #     print(f"{idx+1}. {file}")
    # train_file = input("File number to load as training, or enter 'y' to use train file name: ")
    # if(train_file.isnumeric() and int(train_file) <= len(model_files)):
    #     data_output_file = model_dir + model_files[int(train_file)-1] + ".model"
    # else:
    #     if(train_file == 'y'):
    #         data_output_file = model_dir + data_loader.get_chosen_files()[0] + '.model'
    #     else:
    #         data_output_file = model_dir + train_file + ".model"
    #     if(data_output_file[-6:] != ".model"):
        with open(data_output_file, 'wb') as f:
            pickle.dump(data_package, f)