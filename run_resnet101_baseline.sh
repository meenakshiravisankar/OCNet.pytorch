# check the enviroment info
nvidia-smi
PYTHON="python3"

#network config
NETWORK="resnet101"
METHOD="baseline"
DATASET="idd_train"
NUM_CLASSES=26

#training settings
LEARNING_RATE=5e-3
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=80000
BATCHSIZE=4
INPUT_SIZE='769,769'
USE_CLASS_BALANCE=True
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0
USE_VAL_SET=False
USE_EXTRA_SET=False

#mlflow config
EXPERIMENT_NAME="$1"

# replace the DATA_DIR with your folder path to the dataset.
DATA_DIR='./dataset/idd'
DATA_LIST_PATH='./dataset/list/idd/train_1.lst'
RESTORE_FROM='./pretrained_model/resnet101-imagenet.pth'


#Set the Output path of checkpoints, training log.
TRAIN_LOG_DIR="./log/log_train"
TRAIN_LOG_FILE="./log/log_train/log_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}"
CHECKPOINT_DIR="./checkpoint/"
SNAPSHOT_DIR="./checkpoint/snapshots_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

# Create log and checkpoint directories
mkdir -p ${TRAIN_LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
########################################################################################################################
#  Training
########################################################################################################################
$PYTHON -u train.py --num-classes $NUM_CLASSES --experiment-name $EXPERIMENT_NAME --network $NETWORK --method $METHOD --random-mirror --random-scale --gpu 0,1,2,3 --batch-size $BATCHSIZE \
  --snapshot-dir $SNAPSHOT_DIR  --num-steps $MAX_ITERS --ohem $USE_OHEM --data-list $DATA_LIST_PATH --weight-decay $WEIGHT_DECAY \
  --input-size $INPUT_SIZE --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --use-val $USE_VAL_SET --use-weight $USE_CLASS_BALANCE \
  --snapshot-dir $SNAPSHOT_DIR --restore-from $RESTORE_FROM --start-iters $START_ITERS --learning-rate $LEARNING_RATE  \
  --use-extra $USE_EXTRA_SET --dataset $DATASET --data-dir $DATA_DIR  > $TRAIN_LOG_FILE 2>&1

