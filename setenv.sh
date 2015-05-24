# Script to setup enviroment variables
# Usage: . setenv.sh OR source setenv.sh

# Get project dir
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# Add caffe-tnarihi
export CAFFE_ROOT=$DIR/3rdparty/caffe
# Add source to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$DIR/src:$CAFFE_ROOT/python:$DIR/3rdparty/caffe-helper/python
# Active virtual enviroment
if [ -d "$DIR/venv" ]; then
  source venv/bin/activate
fi
