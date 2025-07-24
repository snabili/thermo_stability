from sklearn.inspection import partial_dependence
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score

import os, os.path as osp, logging, re, time, json
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import argparse, sys
import warnings
from collections import defaultdict
np.random.seed(1001)

# thermo_stability/logger.py
import os
import logging
import warnings

def setup_logging(name='thermo_logger', log_path=None, level=logging.INFO, console=True):
    """
    Create a reusable logger with both file and console outputs.

    Args:
        name (str): Logger name.
        log_path (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        console (bool): Whether to log to stdout.
    
    Returns:
        logging.Logger: Configured logger object.
    """
    # Suppress noisy warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid double logging

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    if logger.hasHandlers():
        logger.handlers.clear()
    # Avoid duplicate handlers
    if not logger.handlers:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    logger.info("Logger initialized successfully.")
    return logger

logger = setup_logging()


def pull_arg(*args, **kwargs):
    """
    Pulls specific arguments out of sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args

def read_arg(*args, **kwargs):
    """
    Reads specific arguments from sys.argv but does not modify sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, _ = parser.parse_known_args()
    return args


#from contextlib import contextmanager
# decorator as command dispatcher
#@contextmanager
class Scripter: # --> scripter decorator
    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg('script', choices=list(self.scripts.keys())).script
        #setup_logging().info('Running %s', script)
        logger.info('Running %s', script)
        self.scripts[script]()


@contextmanager
def time_and_log(begin_msg, end_msg='Done'):
    try:
        t1 = time.time()
        logger.info(begin_msg)
        yield None
    finally:
        t2 = time.time()
        nsecs = t2-t1
        nmins = int(nsecs//60)
        nsecs %= 60
        logger.info(end_msg + f' (took {nmins:02d}m:{nsecs:.2f}s)')

def imgcat(path):
    """
    Only useful if you're using iTerm with imgcat on the $PATH:
    Display the image in the terminal.
    """
    os.system('imgcat ' + path)


def set_matplotlib_fontsizes(smaller=14,small=18, medium=22, large=26):
    import matplotlib.pyplot as plt
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=medium)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title

def plot_roc(fpr, tpr, auc_val, label, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_metrics(par, train_acc, val_acc, train_f1, val_f1,
                       train_auc, val_auc,xlab, filename):
    """
    Plots Accuracy, F1 Score, and AUC vs. hypertune pars
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Create 3 subplots horizontally
    label = {0: 'Accuracy', 1:'F1-score', 2:'AUC'}
    metric = {0: [train_acc, val_acc], 1:[train_f1, val_f1], 2:[train_auc, val_auc]}
    for i in range(3):
        axs[i].plot(par, metric[i][0], label='Train ' + label[i],     marker='o')
        axs[i].plot(par, metric[i][1], label='Validation '+ label[i], marker='o')
        #axs[i].set_xscale('log')
        #axs[i].set_yscale('log')
        axs[i].set_xlabel(xlab)
        axs[i].set_ylabel(label[i])
        axs[i].set_title(label[i]+' vs '+xlab)
        axs[i].legend()
        axs[i].grid(True)
    fig.tight_layout()
    save_path = os.path.join(plotpath, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f'Plot saved to {save_path}')

def save_results(path, ml_type, grid_result):
    """ 
    works for LR and RF not DNN
    DNN is intensive to run all hyperpars at once
    to run it in one go from main directory: 
	bash /script/rundnn_hypertunes.sh 

    """
    hypertune_file = os.path.join(path, 'MLHypertune_pars', ml_type + '_hypertune.txt')
    os.makedirs(os.path.dirname(hypertune_file), exist_ok=True)
    print(hypertune_file)

    if not os.path.exists(hypertune_file):
        with open(hypertune_file, 'w') as f:
            f.write(ml_type + " Best Score: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
        logger.info("Created " + ml_type + "_hypertune.txt.")
    else:
        with open(hypertune_file, 'a') as f: # to update hypertune file
            f.write(ml_type + " Best Score: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
        logger.info(ml_type + "_hypertune.txt already exists.")

def extract_best_scores(filename):
    pattern = r"scoretune\s+-\s+BS=(\d+),\s+HU=(\d+),\s+NL=(\d+)\s+→\s+mean score=([\d.]+)\s+and error=([\d.]+)"
    results = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                bs, hu, nl, score, error = match.groups()
                results.append((score, bs, hu, nl))
    return results

#def compute_pdp(feat):
def compute_pdp(feat, wrapped_model, xpd_train_scaled):
    return feat, partial_dependence(
        wrapped_model,
        xpd_train_scaled,
        features=[feat],
        kind='average',
        grid_resolution=100,
    )


def metric_val(model, xtrain, xval, ytrain, yval):
    ML_train = defaultdict(list)
    ML_val = defaultdict(list)
    MLT, MLV = model.predict(xtrain), model.predict(xval)
    ML_train['acc'], ML_val['acc'] = accuracy_score(ytrain, MLT), accuracy_score(yval, MLV)
    ML_train['f1'],  ML_val['f1']  = f1_score(ytrain, MLT), f1_score(yval, MLV)
    ML_train['auc'], ML_val['auc'] = roc_auc_score(ytrain, MLT), roc_auc_score(yval, MLV)

    return ML_train, ML_val

def metric_dict(train, val):
    ml_train, ml_val = defaultdict(list), defaultdict(list)
    key = ['acc','f1','auc']
    ml_train, ml_val  = defaultdict(list), defaultdict(list)
    for d,v in enumerate(list(train.values())):
        for k in key:
            ml_train[k].append(v[k])
    for d,v in enumerate(list(val.values())):
        for k in key:
            ml_val[k].append(v[k])
    return ml_train, ml_val

def none_or_int(val):
    return None if val == 'None' else int(val)
