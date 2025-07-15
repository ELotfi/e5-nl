export WANDB_MODE=disabled

    # --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    # --query_instruction_format '{}{}' \
	# --kd_loss_type kl_div \
	#    --deepspeed ds_stage0.json \
	#    --negatives_cross_device \
	#    --gradient_checkpointing \



train_data="\
    /home/ubuntu/projects/RetData/FlagEmbedding/examples/finetune/embedder/example_data/retrieval \
    /home/ubuntu/projects/RetData/FlagEmbedding/examples/finetune/embedder/example_data/sts/sts.jsonl \
    /home/ubuntu/projects/RetData/FlagEmbedding/examples/finetune/embedder/example_data/classification-no_in_batch_neg \
    /home/ubuntu/projects/RetData/FlagEmbedding/examples/finetune/embedder/example_data/clustering-no_in_batch_neg "

# set large epochs and small batch size for testing
num_train_epochs=4
per_device_train_batch_size=2

# set num_gpus to 2 for testing
num_gpus=1
CUDA_VISIBLE_DEVICES=1

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path intfloat/multilingual-e5-base \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 4 \
    --query_max_len 64 \
    --passage_max_len 400 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir ./test_encoder_only_multilingual-e5-base \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 1000 \
    --temperature 0.02 \
	--negatives_cross_device \
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
