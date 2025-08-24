export WANDB_MODE=disabled

# train_data="\
#     ../example_data/retrieval \
#     ../example_data/sts/sts.jsonl \
#     ../example_data/classification-no_in_batch_neg \
#     ../example_data/clustering-no_in_batch_neg "
    # --output_dir ./test_encoder_only_base_bge-large-en-v1.5_sd \
    # --overwrite_output_dir \
    # --deepspeed ../../ds_stage0.json \
    # --kd_loss_type kl_div \
	# --query_instruction_for_retrieval 'query: ' \
	# --passage_instruction_for_retrieval 'passage: ' \
    # --query_instruction_format '{}{}' \
	# --passage_instruction_format '{}{}' \
	# last_token
	#    --negatives_cross_device \


num_train_epochs=3
per_device_train_batch_size=64
num_gpus=1
model_name_or_path="FremyCompany/roberta-base-nl-oscar23"  # Qwen/Qwen3-Embedding-4B
hf_hub_token=''

python build_data.py  --use_old_data True --model $model_name_or_path --token $hf_hub_token #--use_cnv_data False --model $model_name_or_path --token $hf_hub_token --is_llm False 

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
	--add_lora False \
	--lora_rank 16 \
	--lora_alpha 32 \
	--lora_dropout 0.05 \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 96 \
    --passage_max_len 450 \
    --pad_to_multiple_of 8 \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
	--query_instruction_for_retrieval 'query: ' \
	--passage_instruction_for_retrieval 'passage: ' \
    --query_instruction_format '{}{}' \
	--passage_instruction_format '{}{}' \
	--knowledge_distillation True \
"

training_args="\
    --learning_rate 2e-5 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
	--gradient_accumulation_steps 2 \
	--gradient_checkpointing False \
	--negatives_cross_device False \
    --dataloader_drop_last True \
    --warmup_steps 2000 \
	--weight_decay 0.1 \
    --logging_steps 10 \
    --save_total_limit 4 \
    --save_strategy steps \
    --save_steps 0.166 \
	--push_to_hub True \
	--hub_model_id  Ehsanl/Robbert_base23_old_7neg_kd \
	--hub_token $hf_hub_token \
    --temperature 0.02 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
"

cmd="torchrun --nproc_per_node $num_gpus \
    finetune.py \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
