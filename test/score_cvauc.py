import re, os,sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from thermo_stability import config, utils

filepath = config.FILE_DIR
logpath  = config.LOG_DIR
plotpath = config.PLOT_DIR

parser = argparse.ArgumentParser(description="Hyperparameter tuning for DNN")
# Add arguments
parser.add_argument("--dnnmetric", type=str, help="dnn metric: roc_auc or accuracy",   default=os.path.join(logpath, 'dnn_auc'))
args = parser.parse_args()

hyparpath=os.path.join(logpath,args.dnnmetric)

logger   = utils.setup_logging(name='scoretune',log_path=filepath + "/MLHypertune_pars/DNN_scoretune_"+args.dnnmetric+".txt")
utils.set_matplotlib_fontsizes()

# Dictionary: {(BS, HU): {NL: [scores...]}}
grouped_scores = defaultdict(lambda: defaultdict(list))

pattern = (
    r"'batch_size':\s*(\d+).*?"
    r"'model__hidden_units':\s*(\d+).*?"
    r"'model__layer_num':\s*(\d+).*?"
    r"'mean_test_score':\s*array\(\[([\d.]+)\]\).*?"
    r"'std_test_score':\s*array\(\[([\d.]+)\]\)"
)

for filename in os.listdir(hyparpath):
    file_path = os.path.join(hyparpath, filename)
    if not os.path.isfile(file_path):
        continue
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                bs, hu, nl, score, error = match.groups()
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

# Plotting parameters
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

plt.xticks(ticks=x, rotation=30, ha="right",fontsize=14, labels=group_labels)
plt.yticks(fontsize=14)
plt.ylabel("Mean CV Score",fontsize=16)
plt.ylim(min(h)*0.99,max(h)*1.01)
plt.title("Mean Cross-Validation Scores vs DNN Hyperparameters",fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(fontsize=16,ncol=3)
plt.tight_layout()
plt.savefig(plotpath + f'/hyperparameters_cvscores_'+args.dnnmetric+'.pdf')


# save highest score into log file to use in ultimate classification
best_score = -float('inf')
best_bs_hu = None
best_nl = None

for bs_hu_key, nl_dict in grouped_scores.items():
    for nl_key, (score, error) in nl_dict.items():
        if score > best_score:
            best_score = score
            best_bs_hu = bs_hu_key
            best_nl = nl_key

logger.info(f" Highest Score: {best_score:.8f}")
logger.info(f" Best Params: {best_bs_hu}, {best_nl}")

