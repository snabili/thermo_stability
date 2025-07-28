import re, os,sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from thermo_stability import config, utils

filepath = config.FILE_DIR
logpath  = config.LOG_DIR
plotpath = config.PLOT_DIR

logger   = utils.setup_logging(name='scoretune',log_path=filepath + "/TEST.txt")
hyperpath=os.path.join(logpath,'dnn_f1score')

# Create argument parser
parser = argparse.ArgumentParser(description="barcharts NL fixed")
# Add arguments
parser.add_argument("--metric",     type=str,   help="metric to plot",    default='roc') # options: roc, acc, f1
args = parser.parse_args()

utils.set_matplotlib_fontsizes()

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

def extract_key(filname):
    bs = int(re.search(r'bs-(\d+)', filname).group(1))
    hd = int(re.search(r'hd-(\d+)', filname).group(1))
    nl = int(re.search(r'nl-(\d+)', filname).group(1))
    return (bs, hd, nl)

def dict_prep(pattern, hyperpath, metric):
    if metric=='roc': ext = [3,4]
    elif metric == 'f1': ext = [5,6]
    else: ext = [7,8]
    grouped_scores = defaultdict(lambda: defaultdict(list))
    fnames = os.listdir(hyperpath)
    sorted_files = sorted(fnames, key=extract_key)
    for filename in sorted_files:
        bs, hu, nl = extract_key(filename)
        file_path = os.path.join(hyperpath, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as f:
            for line in f:
                match = re.search(pattern, line, re.DOTALL)
                if match:
                    extracted = match.groups()
                    score, error = extracted[ext[0]], extracted[ext[1]]
                    key = (f"BS={bs}", f"HU={hu}")
                    grouped_scores[key][f"NL={nl}"] = [float(score), float(error)]
                    logger.info(f"BS={bs}, HU={hu}, NL={nl} → mean score={score} and error={error}")

    group_labels, nl_labels = [], []
    mean_score,   std_score = [], []
    for (bs_label, hu_label), nl_dict in grouped_scores.items():
        group_labels.append(f"{bs_label}, {hu_label}")
        nl_keys = sorted(nl_dict.keys(), key=lambda x: int(x.split('=')[1]))
        nl_labels = nl_keys  # Assumes all groups have same NLs
        mean_score.append([grouped_scores[(bs_label, hu_label)][nl][0] for nl in nl_keys])
        std_score.append([grouped_scores[(bs_label, hu_label)][nl][1] for nl in nl_keys])
    return mean_score, std_score, group_labels, nl_labels

mean_score, std_score, group_labels, nl_labels = dict_prep(pattern, hyperpath, args.metric)

#print(score, std)

num_groups = len(group_labels)
num_nls = len(nl_labels)
x = np.arange(num_groups)
group_width = 1.0
bar_width = group_width / num_nls

# Plot grouped bar chart
h = [] # to set y-axis limit
plt.figure(figsize=(12, 8))
for i, nl in enumerate(nl_labels):
    bar_positions = x - group_width/2 + i * bar_width + bar_width / 2
    heights = [mean_score[group_idx][i] for group_idx in range(num_groups)]
    h.extend(heights)
    errors  = [std_score[group_idx][i] for group_idx in range(num_groups)]
    plt.bar(bar_positions, heights, width=bar_width, label=nl, yerr=errors)

plt.xticks(ticks=x, rotation=60, ha="right",fontsize=11, labels=group_labels)
plt.yticks(fontsize=14)
plt.ylabel("Mean CV Score",fontsize=16)
plt.ylim(min(h)*0.99,max(h)*1.01)
plt.title(f"Mean CV Scores vs DNN Hyperpars Metric: {args.metric.upper()}",fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(fontsize=14,ncol=3)
plt.tight_layout()
savfilename = os.path.join(plotpath, f'{args.metric.upper()}_hyperpars_barchart.pdf')
plt.savefig(savfilename)
