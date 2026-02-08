# Emotion-Context Motion Generator (4-DoF Arm MVP)

本仓库提供一条完整的 MVP pipeline：
`gen_fake_csv -> make_manifest -> convert_csv_to_parquet -> train_flow -> generate_clip -> eval_basic`

## 安装
```bash
pip install -r requirements.txt
```

## 整条 pipeline 如何运行（从 0 到生成一个动作）
```bash
# 1) 生成假数据（CSV）
python tools/gen_fake_csv.py --out_dir data/raw --num_demos 3 --fps_in 120 --duration 8.0

# 2) 生成 manifest 模板（你可以手工改 emotion/context）
python tools/make_manifest.py --raw_dir data/raw --out data/manifests/dataset.yaml

# 3) CSV -> Parquet + 统计量
python tools/convert_csv_to_parquet.py --manifest data/manifests/dataset.yaml \
  --out_parquet data/processed/clips.parquet --out_stats data/processed/stats.json

# 4) 训练（Flow Matching / Rectified Flow，smoke）
python emg/train/train_flow.py --config configs/train_flow.yaml

# 5) 推理生成 + 后处理
python emg/infer/generate_clip.py --config configs/infer.yaml --out_dir data/tmp/generated --device auto

# 6) 基础评估报告
python emg/train/eval_basic.py --ckpt data/tmp/checkpoints/flow_last.pt \
  --parquet data/processed/clips.parquet --stats data/processed/stats.json
```

## 分阶段测试（逐步验收）
```bash
python tests/stage0_env_test.py
python tests/stage1_data_test.py
python tests/stage2_model_forward_test.py
python tests/stage3_train_smoke_test.py
python tests/stage4_infer_postprocess_test.py
python tests/stage5_quality_metrics_test.py
```

## TensorBoard 实时查看训练曲线
```bash
# 训练时启用 TensorBoard 日志
python emg/train/train_flow.py --config configs/train_flow.yaml --tb_log_dir data/tmp/tensorboard

# 另开一个终端启动 TensorBoard
tensorboard --logdir data/tmp/tensorboard
```

## 训练结果可视化（图形化）
```bash
# 先确保安装 matplotlib
pip install matplotlib

# 训练结束后画 loss 曲线
python tools/plot_training.py --log data/tmp/checkpoints/train_log.json --out data/tmp/plots/loss.png
```

## 说明
- DoF 可配置：模型内部不硬编码 J，来自数据/配置。
- 后处理强制关节位置与速度约束。
- 若本地无 `scipy`，低通滤波会跳过并提示。
- 默认 condition 不只包含 `emotion/task/q0`，还包含 `dq0`、`q_goal` 与 `context_numeric`。
- `tools/make_manifest.py` 会在 manifest 中生成 `context_num_keys`，`convert_csv_to_parquet.py` 会按该配置把 context 投影为固定维度向量。
- 推理时可加 `--strict_task`，当 task 不在训练词表中时直接报错，避免静默回退。
