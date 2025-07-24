import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback # to def class PrintValScore

from sklearn.model_selection import GridSearchCV # to optimize ML hyper-parameters
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# general modules
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime
import psutil
from tqdm import tqdm
from joblib import Parallel
from joblib.parallel import BatchCompletionCallBack
#import resource # to measure cpu time

# costume modules
from thermo_stability import config, utils, processing

scripter  = utils.Scripter()
filepath  = config.FILE_DIR
modelpath = config.MODEL_DIR
logpath   = config.LOG_DIR
logger    = utils.setup_logging(name='hypertune',log_path=logpath + "/dnn_hypertune_f1score.txt")

logger.info('*'*100 + '\n')

from tqdm.auto import tqdm

original_callback = BatchCompletionCallBack

def tqdm_callback(*args, **kwargs):
    tqdm_bar = tqdm(total=args[0].n_dispatched_batches, desc="DNN GridSearchCV", unit="batch")
    cb = original_callback(*args, **kwargs)

    def wrap_batch_completed(self, *a, **kw):
        tqdm_bar.update()
        return cb.batch_completed(*a, **kw)

    cb.batch_completed = wrap_batch_completed.__get__(cb, BatchCompletionCallBack)
    return cb

Parallel._backend_callback = staticmethod(tqdm_callback)

# Create argument parser
parser = argparse.ArgumentParser(description="Hyperparameter tuning for DNN")
# Add arguments
parser.add_argument("--npzfile",    type=str,               help="npz file path",                   default=os.path.join(filepath, 'npz_datasplits.npz'))
parser.add_argument("--HD",         type=int,   nargs="+",  help="tune DNN hidden units")#,         default=[64])
parser.add_argument("--NL",         type=int,   nargs="+",  help="tune number of hidden layers")#,  default=[1])
parser.add_argument("--epochs",     type=int,   nargs="+",  help="tune epochs",                     default=[50])
parser.add_argument("--BS",         type=int,   nargs="+",  help="tune batch size")#,               default=[64])
parser.add_argument("--ml_type",    type=str,               help="ML type to save txt outputfile")
parser.add_argument("--script",     type=str,               help="Name of script function to run (e.g. DNN_hypertune, RF_hypertune, LR_hypertune)")
# Parse the arguments
args = parser.parse_args()

# load features from npz file; --> used scaled features for DNN & LR
X_train,X_train_scaled, y_train = np.load(args.npzfile)['X_train'], np.load(args.npzfile)['X_train_scaled'],np.load(args.npzfile)['y_train']
X_val,  X_val_scaled,   y_val   = np.load(args.npzfile)['X_val'],   np.load(args.npzfile)['X_val_scaled'],  np.load(args.npzfile)['y_val']
X_test, X_test_scaled,  y_test  = np.load(args.npzfile)['X_test'],  np.load(args.npzfile)['X_test_scaled'], np.load(args.npzfile)['y_test']

logger.info('load train arrays')

@scripter
def DNN_hypertune():
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003) # optimize
    # 1. Define DNN model architecture
    def build_model(hidden_units=64, layer_num=4, activation='relu'): # used dummy values for hidden_units and layer_num
        # initiate the model
        model = Sequential()
        model.add(Dense(hidden_units, activation=activation, input_shape=[X_train.shape[1]]))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        # Hidden layers (loop-based architecture)
        for _ in range(layer_num):
            model.add(Dense(hidden_units, activation=activation))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        # Compile the model
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    logger.info('defined optimizer')

    # 2. Wrap the model using KerasClassifier
    model = KerasClassifier(model=build_model, epochs=50, verbose=0)
    logger.info('wrapped model with KerasClassifier')

    # 3. Define the correct hyperparameter grid
    dnn_param_grid = {
        'model__hidden_units':  args.HD,  # hidden layers
        'model__layer_num':     args.NL,  # Number of neurons per layers
        'epochs':               args.epochs,
        'batch_size':           args.BS
    }
    logger.info(f"start running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")
    scorer = make_scorer(f1_score, needs_proba=True) # alternative for scoring: roc_auc_score, accuracy_score
    # 4. Run GridSearchCV
    n_jobs = min(3, cpu_count())

    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()

    grid = GridSearchCV(estimator=model, param_grid=dnn_param_grid, scoring=scorer, cv=3, n_jobs=n_jobs, verbose=2)


    # disable GPU usage with tensorflow; apple silicon M1 not compatible
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

    # fit trained model
    grid_result = grid.fit(X_train_scaled, y_train)

    end_cpu_times = process.cpu_times()

    # Calculate CPU time (user + system)
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)

    logger.info(f"\nCPU time used (main process): {cpu_time_used:.2f} seconds with {n_jobs:.1f} cpu_counts")
    logger.info('Cross Validation results:')
    logger.info(grid_result.cv_results_)
    logger.info(pd.DataFrame(grid_result.cv_results_)[['mean_test_score', 'params']])
    logger.info(f"\nBest Score: {grid_result.best_score_:.4f}, using {grid_result.best_params_}")
    logger.info(f"\nend running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")

    # 6. Save best model hyperparameters result to a txt file
    utils.save_results(modelpath, 'DNN',grid_result)

    # Extract best score and parameters
    best_vals = {}
    best_vals['best_score'] = grid_result.best_score_
    best_vals['best_pars'] = grid_result.best_params_

    # Save best vals + pars to a single .npz file
    bs = '-'.join(map(str, args.BS))
    hd = '-'.join(map(str, args.HD))
    nl = '-'.join(map(str, args.NL))
    npz_filename 	= f'DNN_GridSearch_results_BS-{bs}_HD-{hd}_NL-{nl}_accuracy_higherpars.npz'
    npz_hypertunepath 	= os.path.join(filepath, 'MLHypertune_pars', 'npzfiles')
    os.makedirs(npz_hypertunepath, exist_ok=True)
    np.savez(os.path.join(npz_hypertunepath, npz_filename), **best_vals)

    logger.info("Saved DNN classification models successfully! " + '\n')

# **************** RandomForest Classification ************************************
@scripter
def RF_hypertune():
    # Base model with class weighting
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    # Calibrated wrapper
    calibrated_rf = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3)

    # Pipeline (todo: could add preprocessors here)
    pipeline = Pipeline([('calibrated_rf', calibrated_rf)])
    # Hyperparameter grid (passed through to the base estimator inside CalibratedClassifierCV)
    param_grid = {
        'calibrated_rf__base_estimator__n_estimators': [100, 200, 300, 400, 500, 600],
        'calibrated_rf__base_estimator__max_depth': [5, 10, 20, 30,40,50],
        'calibrated_rf__base_estimator__min_samples_split': [2, 5]
    }
    # Use AUC as scoring metric
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    # Grid search
    logger.info(f"start running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=3)
    grid_result = grid.fit(X_train_scaled, y_train)
    logger.info(f"end running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")
    logger.info("RF Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    best_model = grid.estimator
    best_model.fit(X_train, y_train)
    probs = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    logger.info(f"RF Test AUC: {auc:.4f}")
    # Save best model hyperparameters result to a txt file
    utils.save_results(filepath, 'RF',grid_result)

# ********** Logistic Regression *********************************
@scripter
def LR_hypertune():
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'penalty': ['l1', 'l2'],
        'max_iter': [500, 1000, 2000]
    }
    logger.info(f"start running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")
    grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3)
    grid_result = grid.fit(X_train_scaled, y_train)
    logger.info(f"end running grid search at:  {datetime.now().time().strftime('%H:%M:%S')}")
    logger.info("LR Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    best_model = grid.best_estimator_
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    logger.info(f"LR Test AUC: {auc:.4f}")
    utils.save_results(filepath, 'LR',grid_result) # Save best LogisticRegression model hyperparameters result to a txt file

if __name__ == '__main__':  
    scripter.run()
