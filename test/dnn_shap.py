import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thermo_stability import utils
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# data preparation for MLs
from sklearn.utils import class_weight

logpath = config.LOG_DIR
logger = utils.setup_logging(log_path=logpath + "/shap.txt", name="SHAP")

filepath  = config.FILE_DIR
modelpath = config.MODEL_DIR
plotpath  = config.PLOT_DIR

# Load data split in pandas
pd_datasplit = pd.read_csv(os.path.join(filepath, 'df_datasplit.csv'))
Xpd_train_scaled = pd_datasplit[pd_datasplit['split'] == 'train_scaled'].drop(columns=['split','label'])
Xpd_test_scaled  = pd_datasplit[pd_datasplit['split'] == 'test_scaled'].drop(columns=['split','label'])
Xpd_val_scaled   = pd_datasplit[pd_datasplit['split'] == 'val_scaled'].drop(columns=['split','label'])
ypd_train        = pd_datasplit[pd_datasplit['split'] == 'train_label']['label']
ypd_val          = pd_datasplit[pd_datasplit['split'] == 'val_label']['label']
ypd_test         = pd_datasplit[pd_datasplit['split'] == 'test_label']['label']


act='relu'
# Optimizer with learning rate
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
loss = 'binary_crossentropy'

# learning rate scheduler; lowers the learning rate when the validation loss plateaus
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)


# hardcoded hypertunned model:
batch, neurons, layer_num = 128, 128, 1
act = 'relu'
# Define model
DNN_model = Sequential()
# Input layer
DNN_model.add(layers.Dense(neurons, input_shape=[Xpd_train_scaled.shape[1]]))
DNN_model.add(layers.BatchNormalization())
DNN_model.add(layers.Activation(act))
DNN_model.add(layers.Dropout(0.1))
#

# Hidden layers (using a loop)
for _ in range(layer_num):  # 1 hidden layers
    DNN_model.add(layers.Dense(neurons))
    DNN_model.add(layers.BatchNormalization())
    DNN_model.add(layers.Activation(act))
    DNN_model.add(layers.Dropout(0.1)) # typical dropout for moderate to strong regularization

# Output layer
DNN_model.add(layers.Dense(1,activation='sigmoid'))

#Compile the model
DNN_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10, min_delta = 0.001, restore_best_weights = True)
   
# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(ypd_train), y=ypd_train)
class_weights = dict(enumerate(class_weights))
DNN_model.fit(Xpd_train_scaled, ypd_train, validation_data=(Xpd_val_scaled, ypd_val), batch_size=batch, epochs=50, callbacks=[early_stopping,lr_scheduler], class_weight=class_weights) 



# SHAP explanation (using DeepExplainer for TensorFlow/Keras)
explainer = shap.DeepExplainer(DNN_model, Xpd_train_scaled[:100].values)
shap_values = explainer.shap_values(Xpd_test_scaled[:100].values)


shap_values_2d = np.squeeze(shap_values)

# column indices for the selected features
selected_features = list(Xpd_test_scaled.columns[:9]) + list(Xpd_test_scaled.columns[-4:])
selected_indices = [Xpd_test_scaled.columns.get_loc(col) for col in selected_features]

# Get the mean SHAP values
mean_shap = np.abs(shap_values_2d).mean(axis=0)
selected_shap_values = mean_shap[selected_indices]

# Create labeled feature names with SHAP values
feat_imporname = [f"{selected_features[i]}: {selected_shap_values[i]:.3f}" for i in range(len(selected_features))]

# Plot feature importance
shap.summary_plot(shap_values_2d[:, selected_indices], 
                  Xpd_test_scaled[selected_features][:100].values, 
                  plot_type='bar', 
                  feature_names=feat_imporname,
                  show=False)
plt.tight_layout()
plt.xscale('log')
featimpor_filename = os.path.join(plotpath,'dnn_impfeat.pdf')
plt.savefig(featimpor_filename)
plt.close()

# Plot SHAP
shap.summary_plot(shap_values_2d[:, selected_indices], 
		Xpd_test_scaled[selected_features][:100].values,
		feature_names=feat_imporname,
		show=False)
shap_filename = os.path.join(plotpath,'dnn_shap.pdf')
plt.savefig(shap_filename)
plt.close()
