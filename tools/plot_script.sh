#!/bin/bash
PLOT_TOOL_PATH="$CAFFE_ROOT/tools/extra/plot_training_log.py.example"

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ]
then
  echo "Usage: plot_script.sh TYPE OUTPUT LOGFILE"
else
    $PLOT_TOOL_PATH $1 $2 $3
fi
