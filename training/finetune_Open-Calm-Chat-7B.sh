DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export MODEL_NAME=Open-Calm-Chat-7B

export SHOW_DATA=0

BASE_MODEL="${DIR}/../pretrained/open-calm-7b/cyberagent_open-calm-7b/"

TOTAL_STEPS=${FINETUNE_TOTAL_STEPS:-20000}
CHECKPOINT_STEPS=${FINETUNE_CHECKPOINT_STEPS:-100}
CHECKPOINT_PATH=${FINETUNE_CHECKPOINT_PATH:-"${DIR}/../model_ckpts/${MODEL_NAME}"}

DATASETS="\
${DIR}/../data/OIG/files/hhrlhf_evol.jsonl:1,\
${DIR}/../data/OIG/files/osst1.jsonl:1,\
${DIR}/../data/OIG/files/databricks-dolly.jsonl:1\
"

ARGS="--model-name ${BASE_MODEL} \
--tokenizer-name ${BASE_MODEL} \
--project-name together \
--model-type gptneox \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name \
"${DATASETS}" \
--checkpoint-path ${CHECKPOINT_PATH} \
--total-steps ${TOTAL_STEPS} --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps ${CHECKPOINT_STEPS} \
--lr 1e-5 --seq-length 2048 --batch-size 32 --micro-batch-size 1 --gradient-accumulate-step 1 \
--train-log-backend wandb \
--dist-url tcp://127.0.0.1:7033 \
--num-layers 8 --embedding-dim 4096 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)
