# General modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns # --> heat maps
import pickle # --> to access dnn history
import joblib
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ML modules
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import xgboost as xgb # --> feature importance
# custom modules
from thermo_stability import config, utils, processing


scripter = utils.Scripter()

filepath  = config.FILE_DIR
logpath   = config.LOG_DIR
plotpath  = config.PLOT_DIR
modelpath = config.MODEL_DIR

logger = utils.setup_logging(log_path=logpath + "/plotting.txt", name="Plotting")
utils.set_matplotlib_fontsizes() # set up plt format


def data_load():
    ''' 
    scaled feat for plotting DNN and LogisticRegression results
    not scaled for RandomForest
    '''
    npz_split = np.load(filepath + '/npz_datasplits.npz')
    X_test,  X_test_scaled,  y_test  = npz_split['X_test'],  npz_split['X_test_scaled'], npz_split['y_test']
    y_true = y_test

    df_splits = pd.read_csv(filepath + '/df_datasplit.csv')
    Xpd_train_scaled     = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
    Xpd_ebh_train_scaled = df_splits[df_splits['split'] == 'train_ebh_scaled'].drop(columns=['split','label'])
    ypd_train            = df_splits[df_splits['split'] == 'train_label']['label']
    ypd_ebh_train        = df_splits[df_splits['split'] == 'train_ebh_label']['label']

    logger.info(f"Sanity check: {len(Xpd_ebh_train_scaled['band_gap'])}, {len(ypd_ebh_train)}")
    return {
        "Xpd_train_scaled": Xpd_train_scaled,
        "Xpd_ebh_train_scaled": Xpd_ebh_train_scaled,
        "ypd_train": ypd_train,
        "ypd_ebh_train": ypd_ebh_train,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_true": y_true,
    }

def map_feature_name(raw_name):
        try:
            index = int(raw_name[1:])  # Remove 'f' and convert to int
            return feature_names[index]
        except:
            return raw_name  # Fallback if mapping fails

@scripter
def dnn_accuracy():
    # Load history from pickle file
    with open(modelpath + '/dnn_history.pkl','rb') as f:
        history_dnn = pickle.load(f)
    # Access loss, accuracy
    logger.info(history_dnn.keys())  # Check available metrics
    # ********* loss + accuracy plots: train & val datasets *********
    metrics= ['loss', 'accuracy']
    titles = ['Loss', 'Accuracy']
    ylabel_map={'loss': 'Loss', 'accuracy': 'Accuracy'}
    plt.figure(figsize=(6 * len(metrics), 4))
    for i, metric in enumerate(metrics, 1):
        train_metric = history_dnn.get(metric)
        val_metric = history_dnn.get(f"val_{metric}")
        if train_metric is None or val_metric is None: continue
        plt.subplot(1, len(metrics), i)
        plt.plot(train_metric, label='Training')
        plt.plot(val_metric, label='Validation')
        plt.xlabel('Epochs')
        ylabel = ylabel_map.get(metric, metric.capitalize()) if ylabel_map else metric.capitalize()
        plt.ylabel(ylabel)
        title = 'DNN ' + ylabel_map[metric] + ' vs Epochs'
        plt.title(title)
        plt.legend()
        plt.tight_layout()
    filename = plotpath + '/dnn_trainVal.pdf'
    plt.savefig(filename)
    plt.close()
    logger.info(f"Plot saved: {filename}")

@scripter
def roc_curves():
    # Load trained model
    dnn_model = load_model(filepath + '/dnn_classification_allelements.h5')
    LR_model  = joblib.load(filepath + '/models/LogisticRegression_model.joblib')
    RF_model  = joblib.load(filepath + '/models/RandomForest_model.joblib')

    datadict = data_load()
    X_test_scaled, X_test, y_true = datadict['X_test_scaled'], datadict['X_test'], datadict['y_true']

    # Get models
    y_pred_proba_dnn = dnn_model.predict(X_test_scaled).flatten() # keras sequential 
    y_pred_proba_LR  = LR_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_proba_RF  = RF_model.predict_proba(X_test)[:, 1]  # Prob for positive class

    dnn_fpr, dnn_tpr, _ = roc_curve(y_true, y_pred_proba_dnn)
    lr_fpr, lr_tpr, _   = roc_curve(y_true, y_pred_proba_LR)
    rf_fpr, rf_tpr, _   = roc_curve(y_true, y_pred_proba_RF)
    dnn_roc_auc = auc(dnn_fpr, dnn_tpr)
    lr_roc_auc  = auc(lr_fpr, lr_tpr)
    rf_roc_auc  = auc(rf_fpr, rf_tpr)

    utils.plot_roc(dnn_fpr, dnn_tpr, dnn_roc_auc, "DNN", plotpath + '/DNN_ROC.pdf')
    utils.plot_roc(lr_fpr,  lr_tpr,  lr_roc_auc,  "LR",  plotpath + '/LR_ROC.pdf')
    utils.plot_roc(rf_fpr,  rf_tpr,  rf_roc_auc,  "RF",  plotpath + '/RF_ROC.pdf')

    # ROC curves: LR, RF, DNN
    plt.figure(figsize=(6, 5))
    plt.plot(lr_fpr, lr_tpr, label=f"LogisticRegression (AUC = {lr_roc_auc:.2f})")
    plt.plot(rf_fpr, rf_tpr, label=f"RandomForest (AUC = {rf_roc_auc:.2f})")
    plt.plot(dnn_fpr, dnn_tpr, label=f"DNN (AUC = {dnn_roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: LR, RF, DNN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = plotpath + '/AllML_ROCs.pdf'
    plt.savefig(filename)
    logger.info(f"Plot saved: {filename}")

@scripter
def ebh_features():
    datadict = data_load()
    Xpd_ebh_train_scaled, ypd_ebh_train = datadict['Xpd_ebh_train_scaled'], datadict['ypd_ebh_train']
    feature_list = list(Xpd_ebh_train_scaled.columns[:9]) + list(Xpd_ebh_train_scaled.columns[-4:])
    for feat in feature_list:
        fig, ax = plt.subplots(figsize=(6, 5))  # Individual figure for each feature
        h = ax.hist2d(np.squeeze(Xpd_ebh_train_scaled[feat]),np.squeeze(ypd_ebh_train),bins=50,cmap='viridis',norm=LogNorm())
        plt.colorbar(h[3], ax=ax)
        ax.set_xlabel(feat,fontsize=16)
        ax.set_ylabel("EBH",fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        filename = plotpath + f"/{feat}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)  # Free memory
        logger.info(f"Saved {filename}")


@scripter
def pdp_feat():
    data = np.load(filepath + '/pdp_results_average.npz', allow_pickle=True)
    feature_list = data.files

    datadict = data_load()
    Xpd_train_scaled = datadict['Xpd_train_scaled']

    for feature_key in feature_list:
        logger.info(f"Plotting PDP for feature: {feature_key}")
        pdp_entry = data[feature_key].item()
        x_vals = np.squeeze(pdp_entry['grid_values'])
        y_vals = np.squeeze(pdp_entry['average'])

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(x_vals, y_vals, color='blue', label='PDP')
        ax1.set_ylabel('Avg Partial Dependence', color='blue', fontsize=10)
        ax1.set_xlabel(feature_key, fontsize=10)
        ax1.tick_params(axis='y', labelsize=5)
        ax1.set_yscale('log')

        # Histogram on twin axis
        ax2 = ax1.twinx()
        ax2.hist(Xpd_train_scaled[feature_key], bins=30, color='gray', alpha=0.3)
        ax2.set_ylabel('Frequency', color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')#, labelsize=5)
        ax2.set_yscale('log')

        fig.tight_layout()
        fig.suptitle(f'PDP vs {feature_key}', fontsize=14)

        # Save individually
        filename = os.path.join(plotpath, f'pdp_{feature_key}.pdf')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # free memory
 

@scripter
def feat_importance():
    xgbmodel = joblib.load(modelpath + '/XGB_model.joblib')
    importance_dict = xgbmodel.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame(list(importance_dict.items()), columns=['raw_feature', 'importance'])

    datadict = data_load()
    Xpd_train_scaled = datadict['Xpd_train_scaled']
    feature_list = list(Xpd_train_scaled.columns[:9]) + list(Xpd_train_scaled.columns[-4:]) # restricting features to plot all but compositions
    feature_names = Xpd_train_scaled.columns

    importance_df['feature'] = importance_df['raw_feature'].apply(map_feature_name)
    filtered_df = importance_df[importance_df['feature'].isin(feature_list)] # Filter only features in feature_list

    #logger.info(f'importance = {importance_df}')
    plt.figure(figsize=(14, 14))
    sns.barplot(x='feature', y='importance',data=filtered_df.sort_values(by='importance', ascending=False),palette='coolwarm')#, annot_kws={"size": 15})
    plt.xticks(rotation=70)
    plt.axhline(y=0.01,linewidth=5,color='black')
    plt.yscale('log')
    plt.tight_layout()
    plt.title("XGBoost: feature importance",fontsize=22)
    filename = plotpath + '/feature_importance.pdf'
    plt.savefig(filename)
    plt.close()

@scripter
def corr_matrix():
    datadict = data_load()
    Xpd_train_scaled = datadict['Xpd_train_scaled']

    feature_list = list(Xpd_train_scaled.columns[:9]) + list(Xpd_train_scaled.columns[-4:]) # restricting features to plot all but compositions
    selected_df = Xpd_train_scaled[feature_list]
    corr_matrix = selected_df.corr() 
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 15})
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = os.path.join(plotpath, 'correlation_matrix.pdf')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    #data_load()
    scripter.run()
