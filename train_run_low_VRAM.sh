export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCEL_CFG=examples/z_image/model_training/pixel/accelerate_config_offload.yaml

accelerate launch --config_file ${ACCEL_CFG} examples/z_image/model_training/train_L2P.py \
  --dataset_base_path /path/L2P_20k_save_seed/images \
  --dataset_metadata_path /path/L2P_20k_save_seed/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths '[
        [    
            "./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors"
        ],
        [
            "/path/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors", 
            "/path/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors", 
            "/path/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors"
        ]
    ]' \
  --tokenizer_path "/path/Z-Image-Turbo/tokenizer" \
  --save_steps 5000 \
  --learning_rate 5e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/L2P_low_VRAM" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --offload_text_encoder \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 8