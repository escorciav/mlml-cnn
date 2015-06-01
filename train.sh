#!/usr/bin/env sh

CAFFE_PATH=$CAFFE_ROOT/build/tools/caffe

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
  echo -n "Usage: train.sh GPU_ID SOLVER_PROTOTXT NETWORK_PROTOTXT "
  echo "OUTPUT_FILE SNAPSHOT(optional for resume training)"
else
  if [ -z "$5" ]
  then
    echo "Train from scratch"
    $CAFFE_PATH train -gpu $1 -model $3 -solver $2 &>> $4
  else
    if [ -z "$6" ]
    then
      echo "Keep on training"
      $CAFFE_PATH train -gpu $1 -model $3 -solver $2 -snapshot $5 &>> $4
    else
      echo "Finetuning"
      $CAFFE_PATH train -gpu $1 -model $3 -solver $2 -weights $5 &>> $4
    fi
  fi
fi
