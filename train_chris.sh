export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#export OUTPUT_DIR="../../../models/alvan_shivam"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
#export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CLASS_DIR="/home/fox/dreambooth/data/person"
export INSTANCE_DIR="/home/fox/github/data/chris"
export OUTPUT_DIR="/home/fox/dreambooth/models/chris"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --shuffle_after_epoch \
  --use_8bit_adam \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=500 \
  --max_train_steps=2000 \
  --concepts_list="concepts_list_chris.json"
