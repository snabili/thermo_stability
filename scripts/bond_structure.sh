#!/bin/bash

# Get total_items from embedded Python
total_items=$(python3 - <<EOF
import json
from thermo_stability import config
with open(config.STRUCTURES_JSON) as f:
    data = json.load(f)
print(len(data))
EOF
)


SCRIPT_DIR="$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Go one level up to project root
LOG_DIR="$ROOT_DIR/logs/bond_stats" # subdirectory path
FILE_DIR="$ROOT_DIR/files/bond_stats"
mkdir -p "$LOG_DIR" # Create it if it doesn't exist
mkdir -p "$FILE_DIR"

batch_size=$((total_items / 50))
echo "Total items: $total_items"
echo "Batch size: $batch_size"

max_jobs=3
job_count=0

for ((i=0; i<total_items; i+=batch_size)); do
    start=$i
    end=$((i + batch_size))
    log_file="$LOG_DIR/run_from${start}to${end}.log"
    csv_output="$FILE_DIR/df_bond_stats_from${start}to${end}.csv"
    # Skip if output already exists
    if [[ -f "$csv_output" ]]; then
        echo "Skipping batch $start to $end (already processed)"
        continue
    fi

    echo "Launching batch $start to $end"
    echo "log file: $log_file"
    echo "csv output: $csv_output"

    python thermo_stability/feature.py bond_structure "$start" "$end" > "$log_file" 2>&1 &
    ((job_count++))
    if ((job_count % max_jobs == 0)); then
        wait  # Wait for current batch of jobs to finish
        echo "Cooling down..."
        sleep 5
    fi
done

wait  # Final wait to catch any remaining jobs

echo "All batches completed."
