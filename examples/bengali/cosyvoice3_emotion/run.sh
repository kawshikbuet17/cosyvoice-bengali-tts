#!/bin/bash
# Bengali CosyVoice3 emotion fine-tuning staged runner.
#
# Run this file from examples/bengali/cosyvoice3_emotion.

set -e
set -o pipefail

. ./path.sh || exit 1

stage=0
stop_stage=5
dataset_root=/home/kawshik/TTS_Dataset_Ghorer_Bazar
data_dir=data
prefix=bengali_emotion
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B
speaker_id=
speaker_ids=
male_speaker_id=emotion_male
female_speaker_id=emotion_female
min_utterances_per_speaker=1
max_speakers=
val_ratio=0.02
test_ratio=0.02
seed=1234
max_text_chars=250
base_instruct=""
neutral_instruct="Speak neutrally."
convert_to_wav=false
sample_rate=24000
num_processes=10
num_utts_per_parquet=1000
cuda_visible_devices=0
num_workers=2
prefetch=100
train_engine=torch_ddp
dist_backend=nccl
job_id=1986
models="llm"
checkpoint=""

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop_stage) stop_stage="$2"; shift 2 ;;
    --dataset_root) dataset_root="$2"; shift 2 ;;
    --data_dir) data_dir="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --pretrained_model_dir) pretrained_model_dir="$2"; shift 2 ;;
    --speaker_id) speaker_id="$2"; shift 2 ;;
    --speaker_ids) speaker_ids="$2"; shift 2 ;;
    --male_speaker_id) male_speaker_id="$2"; shift 2 ;;
    --female_speaker_id) female_speaker_id="$2"; shift 2 ;;
    --min_utterances_per_speaker) min_utterances_per_speaker="$2"; shift 2 ;;
    --max_speakers) max_speakers="$2"; shift 2 ;;
    --val_ratio) val_ratio="$2"; shift 2 ;;
    --test_ratio) test_ratio="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --max_text_chars) max_text_chars="$2"; shift 2 ;;
    --base_instruct) base_instruct="$2"; shift 2 ;;
    --neutral_instruct) neutral_instruct="$2"; shift 2 ;;
    --convert_to_wav) convert_to_wav=true; shift ;;
    --sample_rate) sample_rate="$2"; shift 2 ;;
    --num_processes) num_processes="$2"; shift 2 ;;
    --num_utts_per_parquet) num_utts_per_parquet="$2"; shift 2 ;;
    --cuda_visible_devices) cuda_visible_devices="$2"; shift 2 ;;
    --num_workers) num_workers="$2"; shift 2 ;;
    --prefetch) prefetch="$2"; shift 2 ;;
    --train_engine) train_engine="$2"; shift 2 ;;
    --dist_backend) dist_backend="$2"; shift 2 ;;
    --job_id) job_id="$2"; shift 2 ;;
    --models) models="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

train_dir=${data_dir}/${prefix}_train
dev_dir=${data_dir}/${prefix}_dev
test_dir=${data_dir}/${prefix}_test

export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Prepare Bengali emotion metadata"
  prep_args=(
    --dataset-root "${dataset_root}"
    --output-dir "${data_dir}"
    --prefix "${prefix}"
    --male-speaker-id "${male_speaker_id}"
    --female-speaker-id "${female_speaker_id}"
    --val-ratio "${val_ratio}"
    --test-ratio "${test_ratio}"
    --seed "${seed}"
    --max-text-chars "${max_text_chars}"
    --min-utterances-per-speaker "${min_utterances_per_speaker}"
    --base-instruct "${base_instruct}"
    --neutral-instruct "${neutral_instruct}"
  )
  if [ -n "${speaker_id}" ]; then
    prep_args+=(--speaker-id "${speaker_id}")
  fi
  if [ -n "${speaker_ids}" ]; then
    prep_args+=(--speaker-ids ${speaker_ids})
  fi
  if [ -n "${max_speakers}" ]; then
    prep_args+=(--max-speakers "${max_speakers}")
  fi
  if [ "${convert_to_wav}" = true ]; then
    prep_args+=(--convert-to-wav --sample-rate "${sample_rate}")
  fi
  python local/prepare_bengali_emotion_data.py "${prep_args[@]}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Extract campplus speaker embeddings"
  for x in "${train_dir}" "${dev_dir}" "${test_dir}"; do
    if [ -s "${x}/wav.scp" ]; then
      ../../../tools/extract_embedding.py --dir "${x}" \
        --onnx_path "${pretrained_model_dir}/campplus.onnx"
    fi
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Extract discrete speech tokens"
  for x in "${train_dir}" "${dev_dir}" "${test_dir}"; do
    if [ -s "${x}/wav.scp" ]; then
      ../../../tools/extract_speech_token.py --dir "${x}" \
        --onnx_path "${pretrained_model_dir}/speech_tokenizer_v3.onnx"
    fi
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Make parquet files"
  for x in "${train_dir}" "${dev_dir}" "${test_dir}"; do
    if [ -s "${x}/wav.scp" ]; then
      mkdir -p "${x}/parquet"
      ../../../tools/make_parquet_list.py \
        --num_utts_per_parquet "${num_utts_per_parquet}" \
        --num_processes "${num_processes}" \
        --src_dir "${x}" \
        --des_dir "${x}/parquet"
    fi
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Check generated training lists"
  test -s "${train_dir}/parquet/data.list"
  test -s "${dev_dir}/parquet/data.list"
  echo "Train data list: ${train_dir}/parquet/data.list"
  echo "Dev data list: ${dev_dir}/parquet/data.list"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Train CosyVoice3 Bengali emotion model(s): ${models}"
  num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')
  for model in ${models}; do
    if [ -n "${checkpoint}" ]; then
      checkpoint_path="${checkpoint}"
    else
      checkpoint_path="${pretrained_model_dir}/${model}.pt"
    fi
    echo "Using checkpoint: ${checkpoint_path}"
    torchrun --nnodes=1 --nproc_per_node="${num_gpus}" \
      --rdzv_id="${job_id}" --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine "${train_engine}" \
      --config conf/cosyvoice3_bengali_emotion.yaml \
      --train_data "${train_dir}/parquet/data.list" \
      --cv_data "${dev_dir}/parquet/data.list" \
      --qwen_pretrain_path "${pretrained_model_dir}/CosyVoice-BlankEN" \
      --onnx_path "${pretrained_model_dir}" \
      --model "${model}" \
      --checkpoint "${checkpoint_path}" \
      --model_dir "$(pwd)/exp/cosyvoice3_bengali_emotion/${model}/${train_engine}" \
      --tensorboard_dir "$(pwd)/tensorboard/cosyvoice3_bengali_emotion/${model}/${train_engine}" \
      --ddp.dist_backend "${dist_backend}" \
      --num_workers "${num_workers}" \
      --prefetch "${prefetch}" \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi
