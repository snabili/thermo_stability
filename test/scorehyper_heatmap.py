import re, os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from thermo_stability import utils, config


logpath = config.LOG_DIR
plotpath = config.PLOT_DIR
hyperpath = os.path.join(logpath,'dnn_f1score')

logger   = utils.setup_logging(name='heatmap',log_path=logpath + "/DNN_heatmap.txt")
utils.set_matplotlib_fontsizes()

# Create argument parser
parser = argparse.ArgumentParser(description="heatmap NL fixed")
# Add arguments
parser.add_argument("--metric",     type=str,   help="metric to plot",    default='roc') # options: roc, acc, f1
parser.add_argument("--NL",         type=str,   help="Num_layers",        default='1')
args = parser.parse_args()


pattern = (
    r"batch_size':\s*(\d+).*?"
    r"'model__hidden_units':\s*(\d+).*?"
    r"'model__layer_num':\s*(\d+).*?"
    r"'mean_test_roc_auc':\s*array\(\[([\d.]+)\]\).*?"
    r"'std_test_roc_auc':\s*array\(\[([\d.]+)\]\).*?"
    r"'mean_test_f1':\s*array\(\[([\d.]+)\]\).*?"
    r"'std_test_f1':\s*array\(\[([\d.]+)\]\).*?"
    r"'mean_test_accuracy':\s*array\(\[([\d.]+)\]\).*?"
    r"'std_test_accuracy':\s*array\(\[([\d.]+)\]\)"
    )
results_roc, results_f1, results_acc = defaultdict(list), defaultdict(list), defaultdict(list)

fnames = os.listdir(hyperpath)
def extract_key(filename):
    bs = int(re.search(r'bs-(\d+)', filename).group(1))
    hd = int(re.search(r'hd-(\d+)', filename).group(1))
    nl = int(re.search(r'nl-(\d+)', filename).group(1))
    return (bs, hd, nl)

sorted_files = sorted(fnames, key=extract_key)

for filename in sorted_files:
    file_path = os.path.join(hyperpath, filename)
    if not os.path.isfile(file_path):
        continue
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(pattern, line, re.DOTALL)
            if match:
                extracted = match.groups()
                key = f"BS={extracted[0]}, HU={extracted[1]}, NL={extracted[2]}"
                results_roc[key].extend([float(val) for val in extracted[3:5]])
                results_f1[key].extend([float(val) for val in extracted[5:7]])
                results_acc[key].extend([float(val) for val in extracted[7:9]])

rows = []
if args.metric == 'roc': 
   results = results_roc
elif args.metric == 'f1': results = results_f1
else: results = results_acc


for k, v in results.items():
    bs, hu, nl = [seg.split('=')[1] for seg in k.split(', ')]
    if nl == args.NL:
        rows.append({'BS': int(bs), 'HU': int(hu), 'score': v[0], 'std': v[1]})
df = pd.DataFrame(rows)
pivot = df.pivot(index='BS', columns='HU', values='score')

plt.figure(figsize=(8, 6))
ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
ax.set_title(f"Score Heatmap: NL={args.NL}, Metric:{args.metric.upper()}")
plt.ylabel("Batch Size (BS)")
plt.xlabel("Hidden Units (HU)")
plt.tight_layout()
filename = os.path.join(plotpath, 'heatmap_NL-' + args.NL + '_' + args.metric + '.pdf')
plt.savefig(filename) 
logger.info(f"Plot saved: {filename}")
