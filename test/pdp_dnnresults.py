import numpy as np
import pandas as pd
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib 
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasClassifier # keras optimizers are not picklable --> it can't be sent to subprocesses using joblib's backend
from sklearn.inspection import partial_dependence

# Custom moduls
from thermo_stability import config, utils

utils.set_matplotlib_fontsizes()
filepath = config.FILE_DIR
modelpath= config.MODEL_DIR
logpath   = config.LOG_DIR
logger = utils.setup_logging(logpath + "/plotting.txt")

'''def compute_pdp(feat):
    return feat, partial_dependence(
        wrapped_model,
        Xpd_train_scaled,
        features=[feat],
        kind='average',
        grid_resolution=100,
    )'''

# Load dnn model and datasplit CSV
dnn_model = load_model(modelpath + '/dnn_classification.h5')
df_splits = pd.read_csv(filepath +"/df_datasplit.csv")

# load scaled trained features and labels
Xpd_train_scaled = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
ypd_train        = df_splits[df_splits['split'] == 'train_label']['label']

print('train feat info,', Xpd_train_scaled.shape, ypd_train.shape)


logger.info(f"start wrapping model with keras:  {datetime.now().time().strftime('%H:%M:%S')}")
wrapped_model = KerasClassifier(model=dnn_model, epochs=50, batch_size=32)
wrapped_model.fit(Xpd_train_scaled, ypd_train)

logger.info(f"start partial dependence:  {datetime.now().time().strftime('%H:%M:%S')}")

feature_list = list(Xpd_train_scaled.columns[:9]) + list(Xpd_train_scaled.columns[-4:])
pdp_results_average = {}
for feat in feature_list:
    feat_name, result = utils.compute_pdp(feat, wrapped_model, Xpd_train_scaled)
    pdp_results_average[feat_name] = result
logger.info(f"done partial dependence:  {datetime.now().time().strftime('%H:%M:%S')}")

# Save to disk
joblib.dump(pdp_results_average, filepath + '/pdp_results_average.pkl')  
np.savez(filepath + '/pdp_results_average.npz', **pdp_results_average) # alternative numpy format
