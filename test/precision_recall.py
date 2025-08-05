import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.datasets import make_classification

from tensorflow.keras.models import load_model

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thermo_stability import config, utils
filepath  = config.FILE_DIR
logpath   = config.LOG_DIR
modelpath = config.MODEL_DIR
plotpath = config.PLOT_DIR
logger = utils.setup_logging(log_path=logpath + "/precision_recall.txt", name="Precision-Recall")

# Load test datadict & model
data_dict = np.load(filepath + '/npz_datasplits.npz', allow_pickle=True)
X_test_scaled,  y_test  = data_dict['X_test_scaled'], data_dict['y_test']

dnn_model = load_model(os.path.join(modelpath,'dnn_classification.h5'))


# Get the raw probability scores for the positive class ---
y_scores = dnn_model.predict(X_test_scaled)

# Compute the precision-recall curve data ---
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)


# Find the optimal threshold based on F1-score ---
thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = [f1_score(y_test, (y_scores >= t).astype(int)) for t in thresholds]
optimal_threshold_index = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_index]
max_f1 = f1_scores[optimal_threshold_index]

logger.info(f"\nOptimal Threshold: {optimal_threshold:.4f}")
logger.info(f"Maximum F1-score at this threshold: {max_f1:.4f}")

# Plot precission-recall curve + F1-score vs. threshold
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(thresholds, f1_scores, marker='.', label='F1-score')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.4f})')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
filename = plotpath + '/dnn_f1score_optimization.pdf'
plt.savefig(filename)
plt.close()
logger.info(f"Plot saved: {filename}")
