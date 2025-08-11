# DNN modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.metrics import F1Score

# data preparation for MLs
from sklearn.utils import class_weight
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


from collections import OrderedDict, defaultdict
import pickle
import joblib

# Costume modules
import os, re, sys
from thermo_stability import utils, processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

#script_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
#print(script_dir)

filepath  = config.FILE_DIR 
logpath   = config.LOG_DIR
modelpath = config.MODEL_DIR

'''filepath  = config.FILE_DIR 
logpath   = config.LOG_DIR
modelpath = config.MODEL_DIR
filepath = os.path.join(script_dir, 'files')
logpath = os.path.join(filepath,'logs')
modelpath = os.path.join(filepath,"models","MLHypertune_pars")'''
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
   #dnn_txtfile = os.path.join(logpath,'scoretune_metric-acc.txt')

   dnn_entries = utils.extract_best_scores(dnn_txtfile)
   Params = params(dnn_entries)
   dnn_max = max(Params)
   best_config = Params[dnn_max]
   batch = best_config['batch_size']
   neurons = best_config['model__hidden_units']
   layer_num = best_config['model__layer_num']
   logger.info(best_config)
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
   DNN_model.add(layers.Dense(neurons, input_shape=[X_train_scaled.shape[1]]))
   DNN_model.add(layers.BatchNormalization())
   DNN_model.add(layers.Activation(act))
   DNN_model.add(layers.Dropout(0.1))
   #
   
   # Hidden layers (using a loop)
   for _ in range(layer_num):
       DNN_model.add(layers.Dense(neurons))
       DNN_model.add(layers.BatchNormalization())
       DNN_model.add(layers.Activation(act))
       DNN_model.add(layers.Dropout(0.1)) # typical dropout for moderate to strong regularization
   
   # Output layer
   DNN_model.add(layers.Dense(1,activation='sigmoid'))
   
   # add f1-score as additional metric; read threshold value from test/precision_recall.py code
   target_label = 'Optimal Threshold: '
   f1score_log = os.path.join(logpath,'precision_recall.txt')
   with open(f1score_log,'r') as file:
      for line in file:
         if line.startswith(target_label):
            parts = line.split(':')
            f1score_threshold = float(parts[1].strip())
            break  # Stop searching after finding the value   
   f1_metric = F1Score(name='f1_score',threshold=f1score_threshold)

   #Compile the model
   DNN_model.compile(optimizer=adam, loss=loss, metrics=['accuracy',f1_metric])
   logger.info(DNN_model.summary())
   
   # to avoid overfitting
   early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10, min_delta = 0.001, restore_best_weights = True)
   
   # Compute class weights
   class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
   class_weights = dict(enumerate(class_weights))
   #logger.info('sanity check: ', Xpd_val_scaled.shape, ypd_val.shape)
   history_dnn = DNN_model.fit(Xpd_train_scaled, ypd_train, validation_data=(Xpd_val_scaled, ypd_val), batch_size=batch, epochs=50, callbacks=[early_stopping,lr_scheduler], class_weight=class_weights) 
                               #callbacks=[early_stopping, lr_scheduler], class_weight=class_weights) # alternatively use: callbacks=[lr_scheduler]
   y_DNN_pred = DNN_model.predict(Xpd_test_scaled)
   labels = np.unique(ypd_test)
   
   final_val = defaultdict(list)
   for s in ['val_loss','loss','val_accuracy','accuracy','f1_score','val_f1_score']:
       final_val[s].append(history_dnn.history[s][-1])

   final_log_info = {}
   for k, v in final_val.items():
      value = v[0]
      if isinstance(value, np.ndarray):
         final_log_info[k] = f"{value.item():.4f}"
      else:
         final_log_info[k] = f"{value:.4f}"

   logger.info(final_log_info)
   
   # Save dnn trained classification + history files
   DNN_model.save(modelpath + '/dnn_classification.h5')
   with open(modelpath + "/dnn_history.pkl", "wb") as f:
       pickle.dump(history_dnn.history, f)

@scripter
def lr_classification():
   data = load_iris()
   lr_txtfile =  modelpath + '/LR_hypertune.txt'
   lr_entries = utils.extract_best_scores(lr_txtfile)
   Params = params(lr_entries)
   # read hyperparamters from classification_hyperpars.py output file
   #lr_params = Params['LR'][max(Params['LR'])]
   lr_params = {'max_iter':100, 'penalty':'l2', 'C':1}
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
   '''params = Params(rf_entries)
   # read hyperparamters from classification_hyperpars.py output file
   rf_params = Params['RF'][max(Params['RF'])]'''
   rf_params = {'calibrated_rf__base_estimator__n_estimators':200, 'calibrated_rf__base_estimator__min_samples_split':2, 'calibrated_rf__base_estimator__max_depth':5}
   n_estimator       = rf_params['calibrated_rf__base_estimator__n_estimators']
   min_samples_split = rf_params['calibrated_rf__base_estimator__min_samples_split']
   max_depth         = rf_params['calibrated_rf__base_estimator__max_depth']
   
   RF_model = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimator, min_samples_split=min_samples_split, max_depth=max_depth, random_state=42 + 2)
   RF_model.fit(X_train, y_train) # Random Forests is scale-invariant --> normalization doesn’t help
   
   y_RF_pred = RF_model.predict(X_test)
   labels = np.unique(y_test)
   logger.info(classification_report(y_test, y_RF_pred, target_names=data.target_names[labels]))
   joblib.dump(RF_model, modelpath + "/RandomForest_model.joblib")


# *************************************************************************************
# *********************** XGBoost *****************************************************
# *************************************************************************************
@scripter
def xgboost():
   xgb_model = xgb.XGBClassifier()
   Xpd_train_scaled = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
   ypd_train        = df_splits[df_splits['split'] == 'train_label']['label']
   xgb_model.fit(Xpd_train_scaled, ypd_train)
   joblib.dump(xgb_model, modelpath + "/XGB_model.joblib")


if __name__ == '__main__':
    scripter.run()

