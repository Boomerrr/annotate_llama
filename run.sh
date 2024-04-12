/home/anaconda3/envs/llama/bin/torchrun --master_port 29506 --nproc_per_node 1 annotate_file_multi-shot.py \
    --ckpt_dir model-7b/ \
    --tokenizer_path model-7b/tokenizer.model \
    --max_seq_len 3072 \
    --max_batch_size 6
