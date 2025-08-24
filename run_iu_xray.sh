python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 1 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223

# 如需启用混合精度训练，请添加 --use_amp 参数
# python main.py \
# --image_dir data/iu_xray/images/ \
# --ann_path data/iu_xray/annotation.json \
# --dataset_name iu_xray \
# --max_seq_length 60 \
# --threshold 3 \
# --batch_size 16 \
# --epochs 1 \
# --save_dir results/iu_xray \
# --step_size 50 \
# --gamma 0.1 \
# --seed 9223 \
# --use_amp
