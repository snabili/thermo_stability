import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# data preparation for MLs
from sklearn.utils import class_weight

from thermo_stability import utils, config

filepath  = config.FILE_DIR
plotpath  = config.PLOT_DIR
logpath   = config.LOG_DIR
modelpath = config.MODEL_DIR

logger = utils.setup_logging(log_path=logpath + "/uncertainties.txt", name="Uncertainty")
utils.set_matplotlib_fontsizes() # set up plt format

npz_split = np.load(filepath + '/npz_datasplits.npz')
X_test_scaled,  y_test  = npz_split['X_test_scaled'], npz_split['y_test']
X_val_scaled,   y_val   = npz_split['X_val_scaled'], npz_split['y_val']
X_train_scaled, y_train = npz_split['X_train_scaled'], npz_split['y_train']


def create_my_dnn_model():
   # hyperpars from classification_hyperpars.py code
   dnn_txtfile = os.path.join(logpath,'scoretune_metric-acc.txt')
   dnn_entries = utils.extract_best_scores(dnn_txtfile)
   Params = utils.params(dnn_entries)
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
   
   # Hidden layers (using a loop)
   for _ in range(layer_num):
       DNN_model.add(layers.Dense(neurons))
       DNN_model.add(layers.BatchNormalization())
       DNN_model.add(layers.Activation(act))
       DNN_model.add(layers.Dropout(0.1)) # typical dropout for moderate to strong regularization
   
   # Output layer
   DNN_model.add(layers.Dense(1,activation='sigmoid'))
   
   #Compile the model
   DNN_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
   
   # to avoid overfitting
   early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10, min_delta = 0.001, restore_best_weights = True)
   
   # Compute class weights
   class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
   class_weights = dict(enumerate(class_weights))
   history_dnn = DNN_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), batch_size=batch, epochs=50, callbacks=[early_stopping,lr_scheduler], class_weight=class_weights) 
   return DNN_model


# compute mean prediction and uncertainties
num_models = 5
models = []
for i in range(num_models):
    model = create_my_dnn_model()
    models.append(model)

all_predictions = []
for model in models:
    # Get probabilities from each model
    probabilities = model.predict(X_test_scaled)
    all_predictions.append(probabilities)

# Convert to a NumPy array for easy calculation
all_predictions = np.array(all_predictions)
logger.info(f'sanity check: prediction shape = {all_predictions.shape}')

mean_predictions = np.mean(all_predictions, axis=0)
std_predictions = np.std(all_predictions, axis=0)


plt.plot(mean_predictions,std_predictions,'.')
plt.axvline(x=0.5,linestyle='dashed',color='red',label='Decision Boundary')
plt.xlabel('Mean Prediction (Probability of Stability)')
plt.ylabel('Standard Deviation (Uncertainty)')
plt.title('Uncertainty vs. Mean Prediction')
plt.legend()
plt.grid(True)
plt.text(0.1,0.05,'High Confidence',color='black',fontweight='bold')
plt.text(0.75,0.05,'High Confidence',color='black',fontweight='bold')
plt.text(0.5,0.4,'High Uncertainty',color='red',fontweight='bold')
plotname = os.path.join(plotpath,'Uncertainty_vs_Prediction.pdf')
plt.savefig(plotname)
logger.info(f"Plot saved: {plotname}")
