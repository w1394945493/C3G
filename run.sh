# todo 评估示例
python -m src.main \
    +evaluation=re10k mode=test \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    test.save_compare=true wandb.mode=online \
    checkpointing.load="checkpoint_path" \
    wandb.name="wandb_name"

export CUDA_VISIBLE_DEVICES=0
python /home/lianghao/wangyushen/Projects/C3G/src/main.py \
    +evaluation=re10k_multiview \
    mode=test \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    dataset.re10k.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    test.save_compare=true \
    wandb.mode=disabled \
    checkpointing.load=/home/lianghao/wangyushen/data/wangyushen/Weights/c3g/gaussian_decoder_multiview.ckpt \
    wandb.name=wandb_name \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/c3g/debug/exp_debug/ \
