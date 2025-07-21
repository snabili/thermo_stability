import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
from multiprocessing import Pool, cpu_count
import psutil
from itertools import product

from thermo_stability import config, utils, processing

scripter = utils.Scripter()
filepath = config.FILE_DIR
logpath  = config.LOG_DIR
logger   = utils.setup_logging(name='hypertune',log_path=logpath + "/dnn_hypertune_mp_accuracy.txt")

logger.info('*'*100 + '\n')


# DNN hyper-pars values to try
nl_values = [1, 2, 4]
bs_values = [32, 64, 128]
hd_values = [32, 64, 128]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT_DIR, "files", "logs", "dnn")
os.makedirs(LOG_DIR, exist_ok=True)

def run_batch(args):
    nl, bs, hd = args
    log_filename = f"run_bs-{bs}_hd-{hd}_nl-{nl}_accuracy.log"
    #output_file = os.path.join(LOG_DIR, log_filename)
    output_file = os.path.join(logpath, log_filename)

    if os.path.exists(output_file):
        print(f"Skipping bs={bs}, hd={hd}, nl={nl} (already processed)")
        return

    print(f"Running NL={nl}, BS={bs}, HD={hd}")
    with open(output_file, "w") as f:
        f.write(f"Running NL={nl}, BS={bs}, HD={hd}\n")
        f.write("-" * 40 + "\n")
        process = subprocess.run([
            "python", "test/classification_hyperpars.py",
            "--script", "DNN_hypertune",
            "--HD", str(hd),
            "--BS", str(bs),
            "--NL", str(nl)
        ], stdout=f, stderr=subprocess.STDOUT)
        f.write("\n")

if __name__ == "__main__":
    max_parallel_jobs = 3  # tweak this based on your system

    # Build all combinations of NL, BS, HD
    param_combinations = list(product(nl_values, bs_values, hd_values))

    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()

    with Pool(processes=max_parallel_jobs) as pool:
        pool.map(run_batch, param_combinations)

    end_cpu_times = process.cpu_times()
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system) # Calculate CPU time (user + system)
    logger.info(f"\nCPU time used (main process): {cpu_time_used:.2f} seconds with {max_parallel_jobs:.1f} parralel jobs")

    logger.info("All jobs completed.")

