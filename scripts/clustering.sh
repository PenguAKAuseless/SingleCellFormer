#!/bin/bash

# Create output and log directories
mkdir -p "output"
mkdir -p "logs"

# Run clustering script in background
nohup python3 scripts/clustering.py > "logs/clustering.log" 2> "logs/clustering.err" &
echo $! > "logs/clustering.pid"
echo "Clustering script started with PID $(cat logs/clustering.pid)"
echo "Logs can be found in logs/clustering.log and logs/clustering.err"
echo "To stop the script, use 'kill $(cat logs/clustering.pid)'"
echo "Make sure to check the logs for any errors or output."
echo "You can monitor the script's progress using 'tail -f logs/clustering.log'"