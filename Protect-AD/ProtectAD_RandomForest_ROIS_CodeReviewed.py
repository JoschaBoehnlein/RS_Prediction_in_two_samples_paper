# -*- coding: utf-8 -*-
"""
Created in 2022, reviewed in January 2023
by authors Kevin Hilbert, Charlotte Meinke & Alice Chavanne
"""


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import statistics
import pickle
import csv
import multiprocessing
import os
import mkl
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from sklearn import set_config
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc
import time
from imblearn.over_sampling import RandomOverSampler 

start_time = time.time()


set_config(working_memory = 100000)
print(multiprocessing.cpu_count())
#print(len(os.sched_getaffinity(0)))

mkl.set_num_threads(1)

PATH_WORKINGDIRECTORY = '/home/hilbertk/prediction_protectad/'
OPTIONS_OVERALL = {'name_model': 'PROTECTAD_RF_ROIS_code_review_final'}
OPTIONS_OVERALL['number_iterations'] = 100
OPTIONS_OVERALL['name_features'] = ['d_feat_demo.txt','d_feat_conn.txt','d_feat_graph.txt']
OPTIONS_OVERALL['abbreviations_features'] = ['_feat_demo','_feat_conn','_feat_graph','_majority_voting','_softmax_voting','_softmax_by_oob','meta_learner_2nd_lvl_logreg','meta_learner_2nd_lvl_RF']
OPTIONS_OVERALL['name_labels'] = 'd_labels.txt'


"""Overall Options"""
OPTIONS_OVERALL['test_size_option'] = 0.2
OPTIONS_OVERALL['threshold_option'] = "mean" 

def prepare_data(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    random_state_seed = numrun
    print('The current run is iteration {}.'.format(numrun))

    for model in range(0,3):

        """
        "Import Data und Labels
        """

        features_import_path = os.path.join(PATH_WORKINGDIRECTORY,'recoded_data',OPTIONS_OVERALL['name_features'][model])
        labels_import_path = os.path.join(PATH_WORKINGDIRECTORY, 'recoded_data', OPTIONS_OVERALL['name_labels'])
        features_import = read_csv(features_import_path, sep="\t", header=0)
        labels_import = read_csv(labels_import_path, sep="\t", header=0)

        features_import = features_import.drop(columns="CONN")
        labels_import = labels_import.drop(columns="CONN")

        """
        "Split train / test sets
        """

        X_train, X_test, y_train, y_test = train_test_split(
                features_import, labels_import, stratify=None, test_size=OPTIONS_OVERALL['test_size_option'], random_state=random_state_seed)
        ros_train = RandomOverSampler(random_state=random_state_seed)
        X_train, y_train = ros_train.fit_resample(X_train, y_train)

        ros_test = RandomOverSampler(random_state=random_state_seed)
        X_test, y_test = ros_test.fit_resample(X_test, y_test)

        y_train= np.array(y_train)
        y_test= np.array(y_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)  
        
        save_cv_option_features_train = OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][model] + '_save_cv_fold_' + str (random_state_seed) + '_features_train.txt'
        save_cv_option_features_test = OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][model] + '_save_cv_fold_' + str (random_state_seed) + '_features_test.txt'
        save_cv_option_labels_train = OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][model] + '_save_cv_fold_' + str (random_state_seed) + '_labels_train.txt'
        save_cv_option_labels_test = OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][model] + '_save_cv_fold_' + str (random_state_seed) + '_labels_test.txt'


        full_path_cv_option = os.path.join(PATH_WORKINGDIRECTORY, 'features_train', save_cv_option_features_train)
        with open(full_path_cv_option, 'w', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(X_train)

        full_path_cv_option = os.path.join(PATH_WORKINGDIRECTORY, 'labels_train', save_cv_option_labels_train)
        with open(full_path_cv_option, 'w', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(y_train)

        full_path_cv_option = os.path.join(PATH_WORKINGDIRECTORY, 'features_test', save_cv_option_features_test)
        with open(full_path_cv_option, 'w', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(X_test)

        full_path_cv_option = os.path.join(PATH_WORKINGDIRECTORY, 'labels_test', save_cv_option_labels_test)
        with open(full_path_cv_option, 'w', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(y_test)


def do_iterations(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    print(numrun)
    random_state_seed = numrun

    load_cv_option_cur_model =  os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', 'current_model.txt')
    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))

    # Import Data und Labels

    full_path_cv_option_features_train = os.path.join(PATH_WORKINGDIRECTORY, 'features_train', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str(random_state_seed) + '_features_train.txt')
    full_path_cv_option_labels_train = os.path.join(PATH_WORKINGDIRECTORY, 'labels_train', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str(random_state_seed) + '_labels_train.txt')
    full_path_cv_option_features_test = os.path.join(PATH_WORKINGDIRECTORY, 'features_test', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str(random_state_seed) + '_features_test.txt')
    full_path_cv_option_labels_test = os.path.join(PATH_WORKINGDIRECTORY, 'labels_test', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str(random_state_seed) + '_labels_test.txt')

    X_train = read_csv(full_path_cv_option_features_train, sep="\s", header=None, engine='python')
    X_test = read_csv(full_path_cv_option_features_test, sep="\s", header=None, engine='python')
    y_train = read_csv(full_path_cv_option_labels_train, sep="\s", header=None, engine='python')
    y_test = read_csv(full_path_cv_option_labels_test, sep="\s", header=None, engine='python')


    # Imputation missing values

    X_train_imputed, X_test_imputed = mean_median_mode_imputation(X_train, X_test, random_state_seed)


    # Scaling

    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train_imputed)
    X_train_imputed_scaled = scaler.transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)


    # Feature Selection

    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)  

    clf_elastic_logregression_features = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, fit_intercept=False, tol=0.0001, max_iter=1000, random_state=random_state_seed)
    sfm = SelectFromModel(clf_elastic_logregression_features, threshold=OPTIONS_OVERALL['threshold_option'])
    sfm.fit(X_train_imputed_scaled, y_train)
    X_train_imputed_scaled_selected = sfm.transform(X_train_imputed_scaled)
    X_test_scaled_imputed_selected = sfm.transform(X_test_imputed_scaled)


    # Random Forrest Analyse

    clf = RandomForestClassifier(n_estimators= 1000, criterion = 'gini', max_features= 'auto', max_depth= None, min_samples_split= 2, min_samples_leaf= 1, bootstrap= True, oob_score=True, random_state=random_state_seed)
    clf = clf.fit(X_train_imputed_scaled_selected, y_train)
    
    y_prediction = np.zeros((len(y_test), 2))
  
    y_prediction[:,0] = clf.predict(X_test_scaled_imputed_selected)
        
    y_prediction[:,1] = y_test[:] 
    
   
    meta_learner_input = np.zeros((len(y_test), 4))
    meta_learner_input[:,0] = y_test[:] 
    meta_learner_input[:,1] = clf.predict(X_test_scaled_imputed_selected)
    meta_learner_input[:,2] = clf.predict_proba(X_test_scaled_imputed_selected)[:,0]
    meta_learner_input[:,3] = clf.predict_proba(X_test_scaled_imputed_selected)[:,1]
    
    meta_learner_input_train = np.zeros((len(y_train), 4))
    meta_learner_input_train[:,0] = y_train[:] 
    meta_learner_input_train[:,1] = clf.predict(X_train_imputed_scaled_selected)
    meta_learner_input_train[:,2] = clf.predict_proba(X_train_imputed_scaled_selected)[:,0]
    meta_learner_input_train[:,3] = clf.predict_proba(X_train_imputed_scaled_selected)[:,1]


    # Results Processing

    # Get importances for each feature
    feature_importances = np.zeros((len(sfm.get_support())))
    counter_features_selected = 0
    for number_features in range(len(sfm.get_support())):
        if sfm.get_support()[number_features] == True:
            feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
            counter_features_selected = counter_features_selected + 1
        else:
            feature_importances[number_features] = 0

    
    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = y_prediction[:,0], y_true = y_prediction[:,1], y_prob = meta_learner_input[:,2], fitted_clf = clf)
    
    save_cv_option_meta_learner_input = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    with open(save_cv_option_meta_learner_input, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(meta_learner_input)

    save_cv_option_meta_learner_input_train = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    with open(save_cv_option_meta_learner_input_train, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(meta_learner_input_train)
        
    save_cv_option_oob_input =  os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_oob_acc.txt')
    with open(save_cv_option_oob_input, 'wb') as AutoPickleFile:
        pickle.dump((oob_accuracy), AutoPickleFile)   


    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value
    

def mean_median_mode_imputation(X_train, X_test, random_state_seed):
    """Missing Values are replaced with mode imputations for binary features, median imputations for ordinary features, and mean imputations for dimensional features"""
    imp_arith = SimpleImputer(missing_values=999, strategy='mean')
    imp_median = SimpleImputer(missing_values=888, strategy='median')
    imp_mode = SimpleImputer(missing_values=777, strategy='most_frequent')
    imp_arith.fit(X_train)
    imp_median.fit(X_train)
    imp_mode.fit(X_train)
    X_train_imputed = imp_arith.transform(X_train)
    X_train_imputed = imp_median.transform(X_train_imputed)
    X_train_imputed = imp_mode.transform(X_train_imputed)  
    X_test_imputed = imp_arith.transform(X_test)
    X_test_imputed = imp_median.transform(X_test_imputed)
    X_test_imputed = imp_mode.transform(X_test_imputed)

    return X_train_imputed, X_test_imputed



def result_metrics_binary(y_pred, y_true, y_prob, fitted_clf = None):
    """Result metrics are calculated"""

    counter_class1_correct = 0
    counter_class2_correct = 0
    counter_class1_incorrect = 0
    counter_class2_incorrect = 0

    # Initialize vector saying whether the prediction is correct
    is_prediction_correct = np.zeros(len(y_pred))

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            is_prediction_correct[i] = 1
            if y_true[i] == 1:
                counter_class1_correct = counter_class1_correct + 1
            else:
                counter_class2_correct = counter_class2_correct + 1
        else:
            is_prediction_correct[i] = 0
            if y_true[i] == 1:
                counter_class1_incorrect = counter_class1_incorrect + 1
            else:
                counter_class2_incorrect = counter_class2_incorrect + 1

    accuracy = np.mean(is_prediction_correct)
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect)
    accuracy_class2 = counter_class2_correct / (counter_class2_correct + counter_class2_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class2) / 2
    
    # Calculate metrics only if a fitted classifier is given
    if fitted_clf is None:
        oob_accuracy = float("nan")
        log_loss_value = float("nan")
    else:
        try: # oob accuracy only for random forest and similar classifiers
            oob_accuracy = fitted_clf.oob_score_
        except:
            oob_accuracy = float("nan")
        log_loss_value = log_loss(y_pred, y_prob, normalize=True)
        
    fpr, tpr, thresholds = roc_curve(y_pred, y_prob)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_pred, y_prob, n_bins=10)
    
   
    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value


def save_scores(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL
    
    # Wenn die result_metrics in einem dictionary gespeichert wären, wäre eine kompaktere Darstellung möglich (loop über dictionary)

    load_cv_option_cur_model =   os.path.join(PATH_WORKINGDIRECTORY,'metalearner_input', 'current_model.txt')
    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_per_round_accuracy.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_accuracy_class1.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class1)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_per_round_accuracy_class2.txt') 
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class2)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_balanced_accuracy.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(balanced_accuracy)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_oob_accuracy.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(oob_accuracy)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_log_loss.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(log_loss_value)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_feature_importances.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerows(feature_importances)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_fpr.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fpr)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_tpr.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tpr)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_tprs.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tprs)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_roc_auc.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(roc_auc)
    
    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_fraction_positives.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fraction_positives)

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] +  '_per_round_predicted_value.txt')
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(mean_predicted_value)

    
        
def list_to_flatlist(input_data):

    accuracy_flat = []
    accuracy_class1_flat = []
    accuracy_class2_flat = []
    balanced_accuracy_flat = []
    oob_accuracy_flat = []
    log_loss_value_flat = []
    feature_importances_flat = np.zeros((len(input_data),len(input_data[0][6])))
    fpr_flat = []
    tpr_flat = []
    tprs_flat = []
    roc_auc_flat = []
    fraction_positives_flat = []
    mean_predicted_value_flat = []

    counter = 0

    for sublist in input_data:
        for itemnumber in range(len(sublist)):
            if itemnumber == 0:
                accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 1:
                accuracy_class1_flat.append(sublist[itemnumber])
            elif itemnumber == 2:
                accuracy_class2_flat.append(sublist[itemnumber])
            elif itemnumber == 3:
                balanced_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 4:
                oob_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 5:
                log_loss_value_flat.append(sublist[itemnumber])
            elif itemnumber == 6:
                feature_importances_flat[counter,:] = sublist[itemnumber]
            elif itemnumber == 7:
                fpr_flat.append(sublist[itemnumber])
            elif itemnumber == 8:
                tpr_flat.append(sublist[itemnumber])
            elif itemnumber == 9:
                tprs_flat.append(sublist[itemnumber])
            elif itemnumber == 10:
                roc_auc_flat.append(sublist[itemnumber])
            elif itemnumber == 11:
                fraction_positives_flat.append(sublist[itemnumber])
            elif itemnumber == 12:
                mean_predicted_value_flat.append(sublist[itemnumber])

        counter = counter + 1

    return accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat


def aggregate_scores(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    load_cv_option_cur_model =  os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', 'current_model.txt')
    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))

    accuracy_min = min(accuracy)
    accuracy_max = max(accuracy)
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    accuracy_class1_min = min(accuracy_class1)
    accuracy_class1_max = max(accuracy_class1)
    accuracy_class1_mean = np.mean(accuracy_class1)
    accuracy_class1_std = np.std(accuracy_class1)
    accuracy_class2_min = min(accuracy_class2)
    accuracy_class2_max = max(accuracy_class2)
    accuracy_class2_mean = np.mean(accuracy_class2)
    accuracy_class2_std = np.std(accuracy_class2)
    balanced_accuracy_min = min(balanced_accuracy)
    balanced_accuracy_max = max(balanced_accuracy)
    balanced_accuracy_mean = np.mean(balanced_accuracy)
    balanced_accuracy_std = np.std(balanced_accuracy)
    oob_accuracy_min = min(oob_accuracy)
    oob_accuracy_max = max(oob_accuracy)
    oob_accuracy_mean = np.mean(oob_accuracy)
    oob_accuracy_std = np.std(oob_accuracy)
    log_loss_value_min = min(log_loss_value)
    log_loss_value_max = max(log_loss_value)
    log_loss_value_mean = np.mean(log_loss_value)
    log_loss_value_std = np.std(log_loss_value)
    feature_importances_min = feature_importances.min(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_max = feature_importances.max(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_mean = feature_importances.mean(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_std = feature_importances.std(axis=0).reshape(1,feature_importances.shape[1])


    number_rounds = len(accuracy)



    savepath_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '.txt')
    f = open(savepath_option, 'w')
    f.write('The scikit-learn version is {}.'.format(sklearn.__version__) +
             '\nNumber of Rounds: ' + str(number_rounds) +
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std) +
             '\nMin feature_importances: ' + str(feature_importances_min) + '\nMax feature_importances: ' + str(feature_importances_max) + '\nMean feature_importances: ' + str(feature_importances_mean) + '\nStd feature_importances: ' + str(feature_importances_std))
    f.close()

    print('Number of Rounds: ' + str(number_rounds) +
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std))


    plt.close('all')
    
    # Plot calibration curve
    # Warum wird hier nicht len(mean_predicted_value) genommen, sondern len(fraction_positives)?
    # PULL REQUEST: bitte umsetzen
    # das wäre dann = np.zeros((len(mean_predicted_value))) , richtig?
    min_mean_predicted_value = np.zeros((len(fraction_positives)))
    max_mean_predicted_value = np.zeros((len(fraction_positives)))
    
    for j in range(len(fraction_positives)):
        min_mean_predicted_value[j] = min(mean_predicted_value[j])
        max_mean_predicted_value[j] = max(mean_predicted_value[j])

    minmin_mean_predicted_value = min(min_mean_predicted_value)
    maxmax_mean_predicted_value = max(max_mean_predicted_value)    
    mean_mean_predicted_value = np.linspace(minmin_mean_predicted_value, maxmax_mean_predicted_value, int((round((maxmax_mean_predicted_value - minmin_mean_predicted_value)*100))))
    fraction_positives_interpolated = np.zeros((len(fraction_positives),int((round((maxmax_mean_predicted_value - minmin_mean_predicted_value)*100)))))

    # Plot calibration per iteration
    for i in range(len(fraction_positives)):
        if i == 0:
            plt.plot(mean_predicted_value[i], fraction_positives[i], lw=1, color = 'grey', marker='.', label = 'Individual Iterations', alpha=0.3)
        else:
            plt.plot(mean_predicted_value[i], fraction_positives[i], lw=1, color = 'grey', marker='.', alpha=0.3)

        fraction_positives_interpolated[i,:] = np.interp(mean_mean_predicted_value, mean_predicted_value[i], fraction_positives[i])

    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Perfectly calibrated', alpha=.7)
    
    # Plot mean calibration 
    mean_fraction_positives_interpolated = np.mean(fraction_positives_interpolated, axis=0)
    plt.plot(mean_mean_predicted_value, mean_fraction_positives_interpolated, color='k', label=r'Mean calibration', lw=2, alpha=1)
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.title('Calibration plots  (reliability curve)')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Count')
    plt.ylabel("Fraction of positives")
    plt.legend(loc="lower right", framealpha = 0.92)

    calibrations_path = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][current_model] + '_calibrations.png')
    plt.savefig(calibrations_path, dpi = 300)
    #plt.show()


    plt.close('all')
    
    # Plot roc per iteration and mean roc
    for i in range(len(fpr)):
        if i == 0:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey', label = 'Individual Iterations', alpha=0.3)
        else:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey', alpha=0.3)

    mean_fpr = np.linspace(0, 1, 100)
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_auc)
    plt.plot(mean_fpr, mean_tpr, color='k', label=r'Mean ROC', lw=2, alpha=1) #plus minus: $\pm$
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='w', label='AUC = %0.2f, SD = %0.2f' % (mean_auc, std_auc), alpha=.001)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.7)



def meta_learner_majority_voting(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    random_state_seed = numrun


    """
    "Import Inputs
    """

    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')


    vote_majority = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_majority_proba = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_majority_proba_2 = np.zeros((len(meta_learner_inputs_demo), 1))

    meta_learner_inputs_demo= np.array(meta_learner_inputs_demo)
    meta_learner_inputs_conn= np.array(meta_learner_inputs_conn)
    meta_learner_inputs_graph= np.array(meta_learner_inputs_graph)


    """
    "Majority Voting
    """

    for y in range(len(meta_learner_inputs_demo)):
        vote_majority[y] = statistics.mode((meta_learner_inputs_demo[y,1], meta_learner_inputs_conn[y,1], meta_learner_inputs_graph[y,1]))
        vote_majority_proba[y] = (meta_learner_inputs_demo[y,1] + meta_learner_inputs_conn[y,1] + meta_learner_inputs_graph[y,1])/3
        vote_majority_proba_2[y] = 1- vote_majority_proba[y]
        
    feature_importances = np.ones((3))
    
    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = vote_majority, y_true = meta_learner_inputs_demo[:,0], y_prob = vote_majority_proba)
    
    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value


def meta_learner_softmax_voting(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL
    
    random_state_seed = numrun


    """
    "Import Inputs
    """

    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')


    vote_softmax_raw = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_softmax_raw_2 = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_softmax = np.zeros((len(meta_learner_inputs_demo), 1))

                          
    meta_learner_inputs_demo= np.array(meta_learner_inputs_demo)
    meta_learner_inputs_conn= np.array(meta_learner_inputs_conn)
    meta_learner_inputs_graph= np.array(meta_learner_inputs_graph)

    """
    "Softmax Voting
    """

    for z in range(len(meta_learner_inputs_demo)):
        vote_softmax_raw[z] = meta_learner_inputs_demo[z,2] + meta_learner_inputs_conn[z,2] + meta_learner_inputs_graph[z,2]
        vote_softmax_raw_2[z] = (meta_learner_inputs_demo[z,3] + meta_learner_inputs_conn[z,3] + meta_learner_inputs_graph[z,3])/3
        if vote_softmax_raw[z] > 1.5:
            vote_softmax[z] = 1
        else:
            vote_softmax[z] = 0
    
    feature_importances = np.ones((3))
    
    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = vote_softmax, y_true = meta_learner_inputs_demo[:,0], y_prob = vote_softmax_raw_2)

    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value

def meta_learner_Softmax_by_oob(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    random_state_seed = numrun

    """
    "Import Inputs
    """

    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')


    full_path_cv_option_oob_input_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_oob_acc.txt')
    with open(full_path_cv_option_oob_input_demo, "rb") as input_file:
        oob_demo = pickle.load(input_file)

    full_path_cv_option_oob_input_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_oob_acc.txt')
    with open(full_path_cv_option_oob_input_conn, "rb") as input_file:
        oob_conn = pickle.load(input_file)

    full_path_cv_option_oob_input_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_oob_acc.txt')
    with open(full_path_cv_option_oob_input_graph, "rb") as input_file:
        oob_graph = pickle.load(input_file)



    vote_softmax_raw = np.zeros((len(meta_learner_inputs_demo), 1))
    threshold = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_softmax_raw_2 = np.zeros((len(meta_learner_inputs_demo), 1))
    vote_softmax = np.zeros((len(meta_learner_inputs_demo), 1))


    meta_learner_inputs_demo= np.array(meta_learner_inputs_demo)
    meta_learner_inputs_conn= np.array(meta_learner_inputs_conn)
    meta_learner_inputs_graph= np.array(meta_learner_inputs_graph)      

    """
    "Softmax Voting by oob scores
    """

    for z in range(len(meta_learner_inputs_demo)):
        vote_softmax_raw[z] = meta_learner_inputs_demo[z,2]*oob_demo + meta_learner_inputs_conn[z,2]*oob_conn + meta_learner_inputs_graph[z,2]*oob_graph
        threshold[z] = 0.5*oob_demo + 0.5*oob_conn + 0.5*oob_graph
        vote_softmax_raw_2[z] = (meta_learner_inputs_demo[z,3]*oob_demo + meta_learner_inputs_conn[z,3]*oob_conn + meta_learner_inputs_graph[z,3]*oob_graph)/(threshold[z]*2)
        if vote_softmax_raw[z] > threshold[z]:
            vote_softmax[z] = 1
        else:
            vote_softmax[z] = 0
    
    feature_importances = np.ones((3))

    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = vote_softmax, y_true = meta_learner_inputs_demo[:,0], y_prob = vote_softmax_raw_2)

    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value

def meta_learner_2nd_level_logreg(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    random_state_seed = numrun


    """
    "Import Inputs
    """
    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')
    
    # Ich fände es gut, wenn die Pfade für die untersch. Daten auch untersch. Namen hätten (z.B. _train_demo vs. _test_demo)
    # Sonst ist es sehr verwirrend (siehe auch nächster Metalearner)
    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')


    meta_learner_input_train_demo= np.array(meta_learner_input_train_demo)
    meta_learner_input_train_conn= np.array(meta_learner_input_train_conn)
    meta_learner_input_train_graph= np.array(meta_learner_input_train_graph)
    meta_learner_inputs_demo= np.array(meta_learner_inputs_demo)
    meta_learner_inputs_conn= np.array(meta_learner_inputs_conn)
    meta_learner_inputs_graph= np.array(meta_learner_inputs_graph)


    meta_learner_input_features_train = np.zeros((len(meta_learner_input_train_demo), 3))
    meta_learner_input_features_train[:,0] = meta_learner_input_train_demo[:,1]
    meta_learner_input_features_train[:,1] = meta_learner_input_train_conn[:,1]
    meta_learner_input_features_train[:,2] = meta_learner_input_train_graph[:,1]
    meta_learner_input_labels_train = np.zeros((len(meta_learner_input_train_demo)))
    meta_learner_input_labels_train = meta_learner_input_train_demo[:,0]

    meta_learner_input_features_test = np.zeros((len(meta_learner_inputs_demo), 3))
    meta_learner_input_features_test[:,0] = meta_learner_inputs_demo[:,1]
    meta_learner_input_features_test[:,1] = meta_learner_inputs_conn[:,1]
    meta_learner_input_features_test[:,2] = meta_learner_inputs_graph[:,1]
    meta_learner_input_labels_test = np.zeros((len(meta_learner_inputs_demo)))
    meta_learner_input_labels_test = meta_learner_inputs_demo[:,0]


    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    clf = clf.fit(meta_learner_input_features_train, meta_learner_input_labels_train)

    y_pred = clf.predict(meta_learner_input_features_test)
    y_true = meta_learner_input_labels_test[:]
    y_prob = clf.predict_proba(meta_learner_input_features_test)[:,1]
    
    feature_importances = np.transpose(clf.coef_)[:,0]

    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = y_pred, y_true = y_true, y_prob = y_prob, fitted_clf = clf)
    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value

def meta_learner_2nd_level_RF(numrun):

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    random_state_seed = numrun


    """
    "Import Inputs
    """

    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_save_cv_fold_' + str (random_state_seed) + '_predictions.txt')
    meta_learner_inputs_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_demo = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_demo = read_csv(full_path_cv_option_features_train_demo, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_conn = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_conn = read_csv(full_path_cv_option_features_train_conn, sep="\s", header=None, engine='python')

    full_path_cv_option_features_train_graph = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_save_cv_fold_' + str (random_state_seed) + '_train_predictions.txt')
    meta_learner_input_train_graph = read_csv(full_path_cv_option_features_train_graph, sep="\s", header=None, engine='python')


    meta_learner_input_train_demo= np.array(meta_learner_input_train_demo)
    meta_learner_input_train_conn= np.array(meta_learner_input_train_conn)
    meta_learner_input_train_graph= np.array(meta_learner_input_train_graph)
    meta_learner_inputs_demo= np.array(meta_learner_inputs_demo)
    meta_learner_inputs_conn= np.array(meta_learner_inputs_conn)
    meta_learner_inputs_graph= np.array(meta_learner_inputs_graph)


    meta_learner_input_features_train = np.zeros((len(meta_learner_input_train_demo), 3))
    meta_learner_input_features_train[:,0] = meta_learner_input_train_demo[:,1]
    meta_learner_input_features_train[:,1] = meta_learner_input_train_conn[:,1]
    meta_learner_input_features_train[:,2] = meta_learner_input_train_graph[:,1]
    meta_learner_input_labels_train = np.zeros((len(meta_learner_input_train_demo)))
    meta_learner_input_labels_train = meta_learner_input_train_demo[:,0]

    meta_learner_input_features_test = np.zeros((len(meta_learner_inputs_demo), 3))
    meta_learner_input_features_test[:,0] = meta_learner_inputs_demo[:,1]
    meta_learner_input_features_test[:,1] = meta_learner_inputs_conn[:,1]
    meta_learner_input_features_test[:,2] = meta_learner_inputs_graph[:,1]
    meta_learner_input_labels_test = np.zeros((len(meta_learner_inputs_demo)))
    meta_learner_input_labels_test = meta_learner_inputs_demo[:,0]


    clf = RandomForestClassifier(n_estimators= 1000, criterion= 'gini', max_features= 'auto', max_depth= None, min_samples_split= 2, min_samples_leaf= 1, bootstrap= True, oob_score=True, random_state=random_state_seed)
    clf = clf.fit(meta_learner_input_features_train, meta_learner_input_labels_train)
    
    y_pred = clf.predict(meta_learner_input_features_test)
    y_true = meta_learner_input_labels_test[:]
    y_prob = clf.predict_proba(meta_learner_input_features_test)[:,1]
    
    feature_importances = clf.feature_importances_
    
    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, thresholds, tprs, roc_auc, fraction_positives, mean_predicted_value = result_metrics_binary(y_pred = y_pred, y_true = y_true, y_prob = y_prob, fitted_clf = clf)
                                                                                                                                                                                                                 
    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value

def integrate_2nd_lvl_results():

    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL, current_model

    balanced_accuracies = np.zeros((OPTIONS_OVERALL['number_iterations'],8))
    load_option_demo = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][0] + '_per_round_balanced_accuracy.txt')
    load_option_conn = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][1] + '_per_round_balanced_accuracy.txt')
    load_option_graph = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][2] + '_per_round_balanced_accuracy.txt') 
    load_option_majority = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][3] + '_per_round_balanced_accuracy.txt')
    load_option_softmax = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][4] + '_per_round_balanced_accuracy.txt')
    load_option_softmax_oob = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][5] + '_per_round_balanced_accuracy.txt')
    load_option_logreg = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][6] + '_per_round_balanced_accuracy.txt')
    load_option_RF = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + OPTIONS_OVERALL['abbreviations_features'][7] + '_per_round_balanced_accuracy.txt')
    balanced_accuracies[:,0] = np.array(np.transpose(read_csv(load_option_demo, header=None)))[:,0]
    balanced_accuracies[:,1] = np.array(np.transpose(read_csv(load_option_conn, header=None)))[:,0]
    balanced_accuracies[:,2] = np.array(np.transpose(read_csv(load_option_graph, header=None)))[:,0]
    balanced_accuracies[:,3] = np.array(np.transpose(read_csv(load_option_majority, header=None)))[:,0]
    balanced_accuracies[:,4] = np.array(np.transpose(read_csv(load_option_softmax, header=None)))[:,0]
    balanced_accuracies[:,5] = np.array(np.transpose(read_csv(load_option_softmax_oob, header=None)))[:,0]
    balanced_accuracies[:,6] = np.array(np.transpose(read_csv(load_option_logreg, header=None)))[:,0]
    balanced_accuracies[:,7] = np.array(np.transpose(read_csv(load_option_RF, header=None)))[:,0]

    save_option = os.path.join(PATH_WORKINGDIRECTORY, 'accuracy', OPTIONS_OVERALL['name_model'] + '_all_balanced_accs_per_iteration.txt')
    np.savetxt(save_option, balanced_accuracies, delimiter=',', fmt='%1.3f', header='Demographics,Connectivity,Graph Metrics,Majority Voting,Sotftmax Voting,Obb weighted Softmax,2nd lvl Logistic Regression,2nd level Random Forest', comments='')

def save_current_model(current_model):

    save_cv_option_cur_model = os.path.join(PATH_WORKINGDIRECTORY, 'metalearner_input', 'current_model.txt')
    with open(save_cv_option_cur_model, 'wb') as AutoPickleFile:
        pickle.dump((current_model), AutoPickleFile)



if __name__ == '__main__':
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    pool = Pool(20)
    runs_list = []
    outcomes = []
    for i in range (OPTIONS_OVERALL['number_iterations']):
        runs_list.append(i)
    pool.map(prepare_data,runs_list)
    for model in range(0,8):
        save_current_model(model)
        if model < 3:
            outcomes[:] = pool.map(do_iterations,runs_list)
            #outcomes[:] = map(do_iterations,runs_list)
        elif model == 3:
            outcomes[:] = pool.map(meta_learner_majority_voting,runs_list)
            #outcomes[:] = map(meta_learner_majority_voting,runs_list)
        elif model == 4:
            outcomes[:] = pool.map(meta_learner_softmax_voting,runs_list)
            #outcomes[:] = map(meta_learner_softmax_voting,runs_list)
        elif model == 5:
            outcomes[:] = pool.map(meta_learner_Softmax_by_oob,runs_list)
            #outcomes[:] = map(meta_learner_Softmax_by_oob,runs_list)
        elif model == 6:
            outcomes[:] = pool.map(meta_learner_2nd_level_logreg,runs_list)
            #outcomes[:] = map(meta_learner_2nd_level_logreg,runs_list)
        elif model == 7:
            outcomes[:] = pool.map(meta_learner_2nd_level_RF,runs_list)
            #outcomes[:] = map(meta_learner_2nd_level_RF,runs_list)
        accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat = list_to_flatlist(outcomes)
        save_scores(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat)
        aggregate_scores(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat)
    integrate_2nd_lvl_results()
    pool.close()
    pool.join()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
