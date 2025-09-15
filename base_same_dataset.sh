export WANDB_MODE=disabled
num_train_epochs=1
per_device_train_batch_size=1024
num_gpus=1
model_name_or_path=""
hf_hub_token=''

python build_data.py  --use_old_data True --use_syn_data True --filter_by_dpn True --model $model_name_or_path --token $hf_hub_token #--use_cnv_data False --model $model_name_or_path --token $hf_hub_token --is_llm False 

train_data="data/"
# set large epochs and small batch size for testing


if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
	--trust_remote_code True \
	--load_bf16 True \
	--use_flash_attention False \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 2 \
    --query_max_len 450 \
    --passage_max_len 500 \
    --pad_to_multiple_of 8 \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
	--query_instruction_for_retrieval 'query: ' \
	--passage_instruction_for_retrieval 'passage: ' \
    --query_instruction_format '{}{}' \
	--passage_instruction_format '{}{}' \
"

training_args="\
    --learning_rate 1e-5 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
	--gradient_accumulation_steps 1 \
	--gradient_checkpointing True \
	--negatives_cross_device False \
    --dataloader_drop_last True \
    --warmup_ratio .25 \
	--weight_decay 0.1 \
    --logging_steps 10 \
    --save_total_limit 4 \
    --save_strategy steps \
    --save_steps 0.25 \
	--push_to_hub True \
	--hub_model_id   \
	--hub_token $hf_hub_token \
    --temperature 0.02 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
	--lr_scheduler_type constant_with_warmup \
	--deepspeed ds_stage3.json \
"

cmd="torchrun --nproc_per_node $num_gpus \
    finetune.py \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
