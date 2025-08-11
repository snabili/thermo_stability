import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from thermo_stability import utils, processing
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


filepath = config.FILE_DIR
logpath  = config.LOG_DIR
logger  = utils.setup_logging(name='data processing',log_path=logpath + "/data.txt")
scaler = StandardScaler()

df = pd.read_csv(os.path.join(filepath, 'df_combined.csv'))

excluded = [
        'energy_above_hull','volume','nsites','is_stable','formula_pretty','total_magnetization',
        'stability_label','composition','structure','material_id','all_elements',
        'Unnamed: 89',
        'He','Ne','Ar','Kr','Xe'
    ]
included = [c for c in df.columns if c not in excluded]

df['stability_label'] = processing.classify_stability(df['energy_above_hull'], threshold=0.05)

# *******************************************************************************************************
# ***************************** Split feat to pandas df *************************************************
# *******************************************************************************************************

logger.info(f"Splitting data (pandas): {datetime.now().time().strftime('%H:%M:%S')}")

Xpd_train,     Xpd_temp,      ypd_train,     ypd_temp     = train_test_split(df[included], df['stability_label'],    test_size=0.2, random_state=123, shuffle=True)
Xpd_ebh_train, Xpd_ebh_temp,  ypd_ebh_train, ypd_ebh_temp = train_test_split(df[included], df['energy_above_hull'],  test_size=0.2, random_state=123, shuffle=True)
Xpd_val,       Xpd_test,      ypd_val,       ypd_test     = train_test_split(Xpd_temp,     ypd_temp,                 test_size=0.5, random_state=124, shuffle=True)

# Normalize
Xpd_train_scaled     = pd.DataFrame(scaler.fit_transform(Xpd_train),       columns=Xpd_train.columns,    index=Xpd_train.index)
Xpd_ebh_train_scaled = pd.DataFrame(scaler.fit_transform(Xpd_ebh_train),   columns=Xpd_ebh_train.columns,index=Xpd_ebh_train.index) # to plot ebh vs features
Xpd_val_scaled       = pd.DataFrame(scaler.transform(Xpd_val),             columns=Xpd_val.columns,      index=Xpd_val.index)
Xpd_test_scaled      = pd.DataFrame(scaler.transform(Xpd_test),            columns=Xpd_test.columns,     index=Xpd_test.index)

# Label features
X_splits = {}
for df_, label in zip(
    [Xpd_train, Xpd_ebh_train, Xpd_train_scaled, Xpd_ebh_train_scaled, Xpd_val, Xpd_val_scaled, Xpd_test, Xpd_test_scaled],
    ['train',  'ebh_train',   'train_scaled',   'train_ebh_scaled',   'val',   'val_scaled',   'test',   'test_scaled']
):
    df_['split'] = label
    X_splits[label] = df_

# Label targets
y_splits = {}
for y_, label in zip(
    [ypd_train, ypd_ebh_train, ypd_val, ypd_test],
    ['train_label', 'train_ebh_label', 'val_label', 'test_label']
):
    df_label = y_.to_frame(name='label')
    df_label['split'] = label
    y_splits[label] = df_label

df_splits = pd.concat(
    list(X_splits.values()) + list(y_splits.values()),
    axis=0
)

cols = df_splits.columns.tolist()
cols.insert(0, cols.pop(cols.index('split')))
df_splits = df_splits[cols]
df_splits.to_csv(filepath + '/df_datasplit.csv', index=False)

# *********************************************
# *********** featu to Numpy  *****************
# *********************************************

logger.info(f"Splitting data (numpy): {datetime.now().time().strftime('%H:%M:%S')}")
X = df[included].values
y = df['stability_label'].values
X_train, X_temp, y_train, y_temp = train_test_split(X,      y,       test_size=0.2, random_state=123)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp,  test_size=0.5, random_state=124)

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

np.savez(filepath + '/npz_datasplits.npz',
         X_train=X_train, X_train_scaled=X_train_scaled, y_train=y_train,
         X_val=X_val, X_val_scaled=X_val_scaled, y_val=y_val,
         X_test=X_test, X_test_scaled=X_test_scaled, y_test=y_test)

logger.info(f"Done! {datetime.now().time().strftime('%H:%M:%S')}")
