# check the enviroment info
nvidia-smi
# pytorch 04
PYTHON="python3"


#network config
NETWORK="resnet101"
METHOD="interlaced_dsn"
DATASET="idd_train"

#training settings
LEARNING_RATE=1e-2
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=40000
BATCHSIZE=8
INPUT_SIZE='720,1280'
USE_CLASS_BALANCE=False
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0
USE_VAL_SET=False
USE_EXTRA_SET=False
NUM_CLASSES=26

#mlflow config
EXPERIMENT_NAME="$1"

# replace the DATA_DIR with your folder path to the dataset.
DATA_DIR='./dataset/idd'
DATA_LIST_PATH='./dataset/list/idd/train.lst'
RESTORE_FROM='./pretrained_model/resnet101-imagenet.pth'

# Set the Output path of checkpoints, training log.
TRAIN_LOG_FILE="./log/log_train/log_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}"	
SNAPSHOT_DIR="./checkpoint/snapshots_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

# testing settings
TEST_USE_FLIP=False
TEST_USE_MS=False
TEST_STORE_RESULT=True
TEST_BATCHSIZE=4
PREDICT_CHOICE='whole'
WHOLE_SCALE='1'
TEST_RESTORE_FROM="mlruns/6/6b0a1c5abfdb4982a272f84ddf333245/artifacts/CS_scenes_40000.pth"

########################################################################################################################
#  Testing
########################################################################################################################
mkdir -p log/log_test
mkdir -p visualize
# validation set
TESTDATASET="idd_train"
TEST_SET="val"
TEST_DATA_LIST_PATH="./dataset/list/idd/val.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

if [ "$2" == "val" ]; then
    $PYTHON -u eval.py --data-dir $DATA_DIR --num-classes $NUM_CLASSES --experiment-name $EXPERIMENT_NAME --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
    --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
    --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1
fi

# training set
TESTDATASET="idd_train"
TEST_SET="train"
TEST_DATA_LIST_PATH="./dataset/list/idd/train.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

if [ "$2" == "train" ]; then
    $PYTHON -u eval.py --data-dir $DATA_DIR --num-classes $NUM_CLASSES --experiment-name $EXPERIMENT_NAME --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
    --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
    --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1
fi

## test set
TEST_STORE_RESULT=True
TESTDATASET="idd_test"
TEST_SET="test"
TEST_DATA_LIST_PATH="./dataset/list/idd/test.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

if [ "$2" == "test" ]; then
    $PYTHON -u generate_submit.py --data-dir $DATA_DIR --num-classes $NUM_CLASSES --experiment-name $EXPERIMENT_NAME --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
    --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
    --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1
fi
