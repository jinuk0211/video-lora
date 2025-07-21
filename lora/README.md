export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog_example"
export OUTPUT_DIR="trained-tlora_dog"
export API_KEY="your-wandb-api-key"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="no" \
  --trainer_type="ortho_lora" \  # choose "lora" in case you want to train Vanilla T-LoRA
  --trainer_class="sdxl_tlora" \
  --num_train_epochs=800 \
  --checkpointing_steps=100 \
  --resolution=1024 \
  --wandb_api_key=$API_KEY \  # remove if you prefer not to log during training 
  --validation_prompts="a {0} lying in the bed#a {0} swimming#a {0} dressed as a ballerina" \  # a string of prompts separated by #
  --num_val_imgs_per_prompt=3 \
  --placeholder_token="sks" \
  --class_name="dog" \
  --seed=0 \
  --lora_rank=64 \
  --min_rank=32 \
  --sig_type="last" \
  --one_image="02.jpg" \ # remove if you prefer full dataset training