#!/bin/bash

echo "Currently running Python jobs (data.py bond_structure):"
echo "----------------------------------------------------------"
printf "%-8s %-8s %-10s %-6s %-6s %s\n" "PID" "PPID" "ELAPSED" "%CPU" "%MEM" "COMMAND"

ps -eo pid,ppid,etime,%cpu,%mem,command | grep "python data.py bond_structure" | grep -v grep | while read pid ppid etime cpu mem cmd; do
    printf "%-8s %-8s %-10s %-6s %-6s %s\n" "$pid" "$ppid" "$etime" "$cpu" "$mem" "$cmd"
done

