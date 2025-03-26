import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import uniform
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from tqdm import tqdm
import os
import warnings
from utils.embedding import expand_embeddings, ENCODERS
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import get_emb_stats, EMBEDDING_EXTENSION
from utils.directories import prepared_dir, models_dir, bin_dir
from utils.embeddings_gen import embed_files, embed_files_codebert

warnings.filterwarnings('ignore')


def calculate_metrics(label, pred):
    label = replace_label(label)
    pred = replace_label(pred)
    acc = accuracy_score(label, pred)
    pre = precision_score(label, pred)
    rec = recall_score(label, pred)
    f1 = f1_score(label, pred)
    human_f1 = f1_score(label, pred, pos_label=1)
    ai_f1 = f1_score(label, pred, pos_label=0)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    # print(tn, fp, fn, tp)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)

    return acc, tpr, tnr, f1, human_f1, ai_f1

def replace_label(labels):
    new_labels = []
    for label in labels:
        if label == 'llm':
            new_labels.append(0)
        elif label == 'human':
            new_labels.append(1)
    
    return new_labels

def generate_new_idx(df):
    idx_list = []

    for i, row in df.iterrows():
        if row['actual label'] == 1:
            idx_list.append(str(i)+'_human')
        elif row['actual label'] == 0:
            idx_list.append(str(i)+'_ai')

    df['idx'] = idx_list
    return df

def get_hyperparameters(estimator):
    hyperparameters = {}
    params = estimator.get_params(deep=False)
    for key, value in params.items():
        hyperparameters[key] = value
    return hyperparameters

def fit_models(model_data, train_path):
    final_models = {}

    train_df = pd.read_pickle(train_path)
    X_train = expand_embeddings(train_df, 'code_embeddings')
    y_train = train_df['actual label']

    tuned_clf = model_data['code_'][0]

    #gradient boosting classifier
    clf = GradientBoostingClassifier()
    all_params = tuned_clf.get_params(deep=False)

    clf.set_params(**all_params)
    clf = clf.fit(X_train, y_train)
    final_models['GradientBoosted'] = clf

    #nn
    nn = MLPClassifier()
    nn = nn.fit(X_train, y_train)
    final_models['NeuralNetwork'] = nn
    return final_models

def test_model_df(final_models: dict, test_df: pd.DataFrame, names: list[str]):
    df_stats = get_emb_stats(test_df)
    print(f'files: [{df_stats}] ===================')

    X_test = expand_embeddings(test_df, 'code_embeddings')
    y_test = test_df['actual label']

    # print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
    for label, model in final_models.items():
        pred = model.predict(X_test)
        acc, tpr, tnr, f1, human_f1, ai_f1 = calculate_metrics(y_test, pred.tolist())
        avg_f1 = (human_f1+ai_f1)/2

        print(f'{label}--> Accuracy: {round(acc, 4)} TPR: {round(tpr, 4)} TNR: {round(tnr, 4)} Human_F1: {round(human_f1, 4)} AI_F1: {round(ai_f1, 4)} Avg_F1: {round(avg_f1, 4)}')
        # for index, prediction in enumerate(pred.tolist()):
        #     print(f"{names[index]}: {prediction}")


def test_model(final_models: dict, test_files: dict[str, str]):
    output = pd.DataFrame(columns=['idx', 'acc', 'tpr', 'tnr', 'f1', 'learning_rate', 'n_estimators', 'max_depth', 'loss', 'criterion'])
    auroc_list, acc_list, tpr_list, tnr_list, human_f1_list, ai_f1_list, f1_list = [], [], [], [], [], [], []
    for file_path, file_name  in test_files.items():
        print(f'{file_name} ===================')
        test_df = pd.read_pickle(file_path)
        output_df = pd.DataFrame(columns=['idx', 'code', 'ast', 'actual label', 'pred'])

        X_test = expand_embeddings(test_df, 'code_embeddings')
        y_test = test_df['actual label']

        # print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
        for label, model in final_models.items():
            pred = model.predict(X_test)
            acc, tpr, tnr, f1, human_f1, ai_f1 = calculate_metrics(y_test, pred.tolist())
            avg_f1 = (human_f1+ai_f1)/2

            print(f'{label}--> Accuracy: {round(acc, 4)} TPR: {round(tpr, 4)} TNR: {round(tnr, 4)} Human_F1: {round(human_f1, 4)} AI_F1: {round(ai_f1, 4)} Avg_F1: {round(avg_f1, 4)}')
            # output = pd.concat([pd.DataFrame({'idx': 'test', 'acc': acc, 'tpr': tpr, 'tnr': tnr, 'f1': avg_f1, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth, 'loss': loss, 'criterion': criterion}, index=[0]), output])
            acc_list.append(acc)
            tpr_list.append(tpr)
            tnr_list.append(tnr)
            human_f1_list.append(human_f1)
            ai_f1_list.append(ai_f1)
            f1_list.append(avg_f1)
        


            # if type == 'ast_':
            #     ast_f1_list.append(avg_f1)
            # elif type == 'combined_':
            #     combined_f1_list.append(avg_f1)
            # elif type == 'code_':
            #     code_f1_list.append(avg_f1)

            output_df['actual label'] = test_df['actual label']
            output_df['pred'] = pred

            # output_df.to_csv(f"")


    avg_acc = round(sum(acc_list)/len(acc_list), 4)
    avg_tpr = round(sum(tpr_list)/len(tpr_list), 4)
    avg_tnr = round(sum(tnr_list)/len(tnr_list), 4)
    avg_human_f1 = round(sum(human_f1_list)/len(human_f1_list), 4)
    avg_ai_f1 = round(sum(ai_f1_list)/len(ai_f1_list), 4)
    avg_f1 = round(sum(f1_list)/len(f1_list), 4)

    
    print()
    print('=== AVERAGE SCORES :===')
    print()
    print(f'--> Accuracy: {avg_acc} TPR: {avg_tpr} TNR: {avg_tnr} Human_F1: {avg_human_f1} AI_F1: {avg_ai_f1} F1: {avg_f1}')
    print()
    return output

split_data_path = ''

emb_types = [ 'code_']
model_type = 'xgb'

if __name__ =="__main__":
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/" + models_dir

    file_loader = DataFileDirectory(file_path, EMBEDDING_EXTENSION,  stat_func=get_emb_stats)
    test_file_loader = DataFileDirectory(file_path, EMBEDDING_EXTENSION, stat_func=get_emb_stats)
    model_loader = DataFileDirectory(model_path, '.file')

    train_file_name = file_loader.get_file('Select a file to load for training')

    model_to_load = model_loader.get_file('Select Model to Load')

    tuned_models = pickle.load(open(model_to_load, 'rb'))
    print(f'Train File: {train_file_name} \nModel File: {model_to_load}')
    print('Fitting Classifiers')
    final_models = fit_models(tuned_models, train_file_name)

    custom = input("custom dataset? (y/n)")

    if(custom != "y"):
        test_file_name = test_file_loader.get_file('Select a file for testing')
        while(test_file_name is not None):
            test_file_name = test_file_loader.get_file('Select other datasets to load, exit to continue to next step. ')
        
        output = test_model(final_models, test_file_loader.get_path_to_file_name_mapping())
    else:
        c_loader = DataFileDirectory(os.path.dirname(os.path.abspath(__file__)) + "/" + bin_dir, '.c')
        file = c_loader.get_file('choose some C files to load. ')
        while(file is not None):
            file = c_loader.get_file('choose some C files to load. ')
        
        device = "cuda"  # for GPU usage or "cpu" for CPU usage

        for index, model in enumerate(ENCODERS):
            print(f"{index}: {model}")
        choice = int(input("Choose Encoder"))
        chosen_encoder = ENCODERS[choice]

        tokenizer = AutoTokenizer.from_pretrained(chosen_encoder, trust_remote_code=True)
        model = AutoModel.from_pretrained(chosen_encoder, trust_remote_code=True).to(device)
        if chosen_encoder == ENCODERS[0]:
            output = embed_files_codebert(c_loader, tokenizer, model)
        else:
            output = embed_files(c_loader, tokenizer, model)

        test_model_df(final_models, output, c_loader.get_chosen_files())


# output.to_csv(f"../../data/test_results.csv")
# ast_f1_list, combined_f1_list, code_f1_list = [], [], []


# ast_avg_f1 = round(sum(ast_f1_list)/len(ast_f1_list), 4)
# combined_avg_f1 = round(sum(combined_f1_list)/len(combined_f1_list), 4)
# code_avg_f1 = round(sum(code_f1_list)/len(code_f1_list), 4)
# print(f'--> AST F1: {ast_avg_f1} Combined F1: {combined_avg_f1} Code F1: {code_avg_f1}')
# print(f'--> AST F1: {ast_avg_f1}')

# print()