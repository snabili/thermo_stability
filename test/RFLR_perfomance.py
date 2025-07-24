import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example: generate/load your dataset
# Replace this with your actual data
from sklearn.datasets import make_classification
import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from thermo_stability import config, utils

scripter  = utils.Scripter()

filepath = config.FILE_DIR
plotpath = config.PLOT_DIR

data = np.load(filepath + '/npz_datasplits.npz', allow_pickle=True)
X_train_scaled, X_train = data['X_train_scaled'], data['X_train']
X_val_scaled,   X_val   = data['X_val_scaled'],   data['X_val']
y_train = data['y_train']
y_val = data['y_val']



parser = argparse.ArgumentParser(description="Hyperparameter tuning for DNN")
# Add arguments
parser.add_argument("--lr_c",           type=float,             nargs="+",  help="LR C values",         default=[1]) # [0.001, 0.01, 0.1, 1, 10, 100]
parser.add_argument("--lr_maxiter",     type=int,               nargs="+",  help="LR maxiter",          default=[500]) # [200, 500, 1000, 2000]
parser.add_argument("--rf_nest",        type=int,               nargs="+",  help="RF nest",             default=[200]) # [100,200, 500, 1000]
parser.add_argument("--rf_maxdepth",    type=utils.none_or_int, nargs="+",  help="RF maxdepth",         default=[5]) # [None, 10, 20]
parser.add_argument("--rf_minsampsplit",type=int,               nargs="+",  help="RF min_samp_split",   default=[2]) # [2, 5]
parser.add_argument("--script",         type=str,                           help="func to run")
args = parser.parse_args()

@scripter
def LR_performance():
    LR_train, LR_val = defaultdict(list), defaultdict(list)
    train, val = defaultdict(list), defaultdict(list)
    ml_var = []
    xlab = str()
    if len(args.lr_c) > 1:
        ml_var = args.lr_c
        xlab = 'C'
        for C in ml_var:
            lr_model = LogisticRegression(C=C, penalty='l2', max_iter=args.lr_maxiter[0], solver='liblinear')
            lr_model.fit(X_train_scaled, y_train)
            LR_train[C], LR_val[C] = utils.metric_val(lr_model,X_train_scaled, X_val_scaled, y_train, y_val)

    else:
        ml_var = args.lr_maxiter
        xlab = 'MaxIter'
        for m in ml_var:
            lr_model = LogisticRegression(C=args.lr_c[0], penalty='l2', max_iter=m, solver='liblinear')
            lr_model.fit(X_train_scaled, y_train)
            LR_train[m], LR_val[c] = utils.metric_val(lr_model,X_train_scaled, X_val_scaled, y_train, y_val)
    
    train, val = utils.metric_dict(LR_train, LR_val)
    utils.plot_metrics(ml_var, train['acc'], val['acc'], train['f1'], val['f1'],train['auc'], val['auc'], xlab=xlab, filename='LR_'+xlab+'.pdf')

@scripter
def RF_performance():
    RF_train, RF_val = defaultdict(list), defaultdict(list)
    train, val = defaultdict(list), defaultdict(list)
    ml_var = []
    xlab = str()
    if len(args.rf_nest) > 1:
        ml_var = args.rf_nest
        xlab = 'NumEstimator'
        for n in ml_var:
            rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=n, min_samples_split=args.rf_minsampsplit[0], max_depth=args.rf_maxdepth[0], random_state=42 + 2)
            rf_model.fit(X_train_scaled, y_train)
            RF_train[n], RF_val[n] = utils.metric_val(rf_model,X_train, X_val, y_train, y_val)

    elif len(args.rf_maxdepth) > 1:
        ml_var = args.rf_maxdepth
        xlab = 'MaxDepth'
        for d in ml_var:
            rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=args.rf_nest[0], min_samples_split=args.rf_minsampsplit[0], max_depth=d, random_state=42 + 2)
            rf_model.fit(X_train_scaled, y_train)
            RF_train[d], RF_val[d] = utils.metric_val(rf_model,X_train, X_val, y_train, y_val)

    else:
        ml_var = args.rf_minsampsplit
        xlab = 'MinSampleSplit'
        for m in ml_var:
            rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=args.rf_nest[0], min_samples_split=m, max_depth=args.rf_maxdepth[0], random_state=42 + 2)
            rf_model.fit(X_train_scaled, y_train)
            RF_train[m], RF_val[m] = utils.metric_val(rf_model,X_train, X_val, y_train, y_val)
    
    train, val = utils.metric_dict(RF_train, RF_val)
    utils.plot_metrics(ml_var, train['acc'], val['acc'], train['f1'], val['f1'],train['auc'], val['auc'], xlab=xlab, filename='RF_'+xlab+'.pdf')





if __name__ == '__main__': 
    scripter.run()