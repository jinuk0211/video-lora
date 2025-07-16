# video-lora
```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config ./processed_data/your_sequence/configs/training.toml


```
