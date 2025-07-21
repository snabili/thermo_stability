# DNN modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# data preparation for MLs
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris


# Logistic regression and random forest packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Feature importance
import xgboost as xgb # --> feature importance

# General modules
import numpy as np
import pandas as pd
import os, re, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import OrderedDict, defaultdict
import pickle
import joblib

# Costume modules
from thermo_stability import config, utils, processing

filepath  = config.FILE_DIR 
logpath   = config.LOG_DIR
modelpath = config.MODEL_DIR
logger = utils.setup_logging(log_path=logpath + "/classification.txt", name="Classification")

scripter = utils.Scripter()

# load split data
data_dict = np.load(filepath + '/npz_datasplits.npz', allow_pickle=True)
X_train,    X_train_scaled, y_train = data_dict['X_train'], data_dict['X_train_scaled'],data_dict['y_train']
X_val,      X_val_scaled,   y_val   = data_dict['X_val'],   data_dict['X_val_scaled'],  data_dict['y_val']
X_test,     X_test_scaled,  y_test  = data_dict['X_test'],  data_dict['X_test_scaled'], data_dict['y_test']

df_splits = pd.read_csv(filepath + '/df_datasplit.csv')
Xpd_train_scaled = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
ypd_train        = df_splits[df_splits['split'] == 'train_label']['label']
Xpd_val_scaled = df_splits[df_splits['split'] == 'val_scaled'].drop(columns=['split','label'])
ypd_val        = df_splits[df_splits['split'] == 'val_label']['label']
Xpd_test_scaled = df_splits[df_splits['split'] == 'test_scaled'].drop(columns=['split','label'])
ypd_test        = df_splits[df_splits['split'] == 'test_label']['label']


def params(entry):
   Params = defaultdict()
   for i,(score, bs, hu, nl) in enumerate(entry):
      Params[float(score)] = {'batch_size': int(bs), 'model__hidden_units': int(hu), 'model__layer_num': int(nl)}
   return Params

@scripter
def dnn_classification():
   dnn_txtfile =  os.path.join(filepath ,'MLHypertune_pars',  'DNN_scoretune_dnn_accuracy.txt')
   dnn_entries = utils.extract_best_scores(dnn_txtfile)
   Params = params(dnn_entries)
   dnn_max = max(Params)
   best_config = Params[dnn_max]
   batch = best_config['batch_size']
   neurons = best_config['model__hidden_units']
   layer_num = best_config['model__layer_num']
   logger.info(f'best score {dnn_max:.3f}, with params: batch_size={batch:.1f}, neurons={neurons:.1f}, layer_num={layer_num:.1f}')

   
   act='relu'
   # Optimizer with learning rate
   adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
   loss = 'binary_crossentropy'
   
   # learning rate scheduler; lowers the learning rate when the validation loss plateaus
   lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
   
   # Define model
   DNN_model = Sequential()
   # Input layer
   DNN_model.add(layers.Dense(neurons, activation=act, input_shape=[X_train_scaled.shape[1]]))
   DNN_model.add(layers.Dropout(0.1))
   DNN_model.add(layers.BatchNormalization())
   
   # Hidden layers (using a loop)
   for _ in range(layer_num):  # 2 hidden layers
       DNN_model.add(layers.Dense(neurons, activation=act))
       DNN_model.add(layers.Dropout(0.3)) # typical dropout for moderate to strong regularization
       DNN_model.add(layers.BatchNormalization())
   
   # Output layer
   DNN_model.add(layers.Dense(1,activation='sigmoid'))
   
   #Compile the model
   DNN_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
   logger.info(DNN_model.summary())
   
   # to avoid overfitting
   early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5, min_delta = 0.001, restore_best_weights = True)
   
   # Compute class weights
   class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
   class_weights = dict(enumerate(class_weights))
   
   #history_dnn = DNN_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), batch_size=batch, epochs=50, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights) # alternatively use: callbacks=[lr_scheduler]
   #y_DNN_pred = DNN_model.predict(X_test_scaled)
   history_dnn = DNN_model.fit(Xpd_train_scaled, ypd_train, validation_data=(Xpd_val_scaled, ypd_val), batch_size=batch, epochs=50, 
                               callbacks=[early_stopping, lr_scheduler], class_weight=class_weights) # alternatively use: callbacks=[lr_scheduler]
   y_DNN_pred = DNN_model.predict(Xpd_test_scaled)
   labels = np.unique(ypd_test)
   
   final_val = defaultdict(list)
   for s in ['val_loss','loss','val_accuracy','accuracy']:
       final_val[s].append(history_dnn.history[s][-1])
   logger.info({k: f"{v[0]:.4f}" for k, v in final_val.items()})
   
   # Save dnn trained classification + history files
   DNN_model.save(modelpath + '/dnn_classification.h5')
   with open(modelpath + "/dnn_history.pkl", "wb") as f:
       pickle.dump(history_dnn.history, f)

@scripter
def lr_classification():
   data = load_iris()
   lr_txtfile =  modelpath + '/LR_hypertune.txt'
   lr_entries = utils.extract_best_scores(lr_txtfile)
   params = Params(lr_entries)
   # read hyperparamters from classification_hyperpars.py output file
   lr_params = Params['LR'][max(Params['LR'])]
   C = lr_params['C']
   max_iter = lr_params['max_iter']
   penalty = lr_params['penalty']
   
   LR_model = LogisticRegression(C=C, penalty=penalty, solver='saga', max_iter=max_iter, class_weight='balanced') # hypertuned
   LR_model.fit(X_train_scaled, y_train)
   y_LR_pred = LR_model.predict(X_test_scaled)
   labels = np.unique(y_test)
   logger.info(classification_report(y_test, y_LR_pred, target_names=data.target_names[labels]))
   joblib.dump(LR_model, modelpath + "/LogisticRegression_model.joblib")

@scripter
def rf_classification():
   data = load_iris()
   rf_txtfile =  modelpath + '/RF_hypertune.txt'
   rf_entries = utils.extract_best_scores(rf_txtfile)
   params = Params(rf_entries)
   # read hyperparamters from classification_hyperpars.py output file
   rf_params = Params['RF'][max(Params['RF'])]
   n_estimator       = rf_params['calibrated_rf__base_estimator__n_estimators']
   min_samples_split = rf_params['calibrated_rf__base_estimator__min_samples_split']
   max_depth         = rf_params['calibrated_rf__base_estimator__max_depth']
   
   RF_model = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimator, min_samples_split=min_samples_split, max_depth=max_depth, random_state=42 + 2)
   RF_model.fit(X_train, y_train) # Random Forests is scale-invariant --> normalization doesn’t help
   
   y_RF_pred = RF_model.predict(X_test)
   labels = np.unique(y_test)
   logger.info(classification_report(y_test, y_RF_pred, target_names=data.target_names[labels]))
   joblib.dump(RF_model, modelpath + "/model/RandomForest_model.joblib")


# *************************************************************************************
# *********************** XGBoost *****************************************************
# *************************************************************************************
@scripter
def xgboost():
   xgb_model = xgb.XGBClassifier()
   Xpd_train_scaled = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
   ypd_train        = df_splits[df_splits['split'] == 'train_label']['label']
   xgb_model.fit(Xpd_train_scaled, y_train)
   joblib.dump(xgb_model, modelpath + "/XGB_model.joblib")


if __name__ == '__main__':
    scripter.run()

