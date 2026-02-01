# 代码结构与关键函数说明（简版）

本文只列出理解 pipeline 必要的文件与核心函数，便于快速阅读。

## 目录结构（核心）
- `configs/`：配置
  - `dataset.yaml`：数据路径与 clip 参数
  - `train_flow.yaml`：模型与训练参数
  - `infer.yaml`：推理条件与后处理约束
- `data/`：数据与输出
  - `raw/`：原始 CSV
  - `manifests/`：`dataset.yaml` 清单
  - `processed/`：`clips.parquet` 与 `stats.json`
  - `tmp/`：训练、推理、评估输出
- `tools/`：数据工具
- `emg/`：核心代码（dataset / model / train / infer）
- `tests/`：阶段测试脚本（stage0~stage5）

## Pipeline 入口脚本
1) 生成假数据：`tools/gen_fake_csv.py`
2) 生成 manifest：`tools/make_manifest.py`
3) CSV->Parquet：`tools/convert_csv_to_parquet.py`
4) 训练：`emg/train/train_flow.py`
5) 推理：`emg/infer/generate_clip.py`
6) 评估：`emg/train/eval_basic.py`

## 关键文件与必要函数

### 1) `tools/gen_fake_csv.py`
- `synth_traj(...)`：合成平滑轨迹（正弦+分段插值）
- `main()`：生成多条 demo CSV

### 2) `tools/make_manifest.py`
- `main()`：扫描 `raw/` 输出 `data/manifests/dataset.yaml` 模板
  - 根据文件名推断 `task_type`
  - 填默认的 `emotion` 与 `context`

### 3) `tools/convert_csv_to_parquet.py`
- `_find_time_col(...)`：识别 time 列
- `_infer_joint_cols(...)`：识别关节列
- `_lowpass_filter(...)`：低通滤波（可选）
- `_resample(...)`：重采样到 `fps_out`
- `_slice_clips(...)`：滑窗切 clip
- `main()`：写出 `clips.parquet` + `stats.json`

### 4) `emg/datasets/clip_dataset.py`
- `ClipDataset.__getitem__`：返回单条样本
  - `x`：训练输入（`delta_q` 或 `q`）
  - `emotion / task_type / q0`：条件
- `_to_array(...)`：兼容 parquet 中嵌套数组
- `build_task_vocab(...)`：`task_type -> id`

### 5) `emg/datasets/collate.py`
- `collate_batch(...)`：把 batch 组装成张量（含 task_id 映射）

### 6) `emg/models/cond.py`
- `ConditionEncoder`：把 `emotion + task_id + q0` 编码为 `cond_vec`

### 7) `emg/models/unet1d.py`
- `UNet1D.forward(...)`：主干网络，输入 `x,t,cond` 输出 `v_pred`

### 8) `emg/models/flow_matching.py`
- `RectifiedFlow.compute_loss(...)`：Flow Matching 的 loss
- `RectifiedFlow.sample(...)`：迭代采样生成轨迹

### 9) `emg/train/train_flow.py`
- `main()`：训练入口
  - 写 `train_log.json`
  - 可选写 TensorBoard

### 10) `emg/infer/postprocess.py`
- `postprocess_clip(...)`：
  - 反归一化
  - `delta_q -> q` 积分
  - 关节限位与速度约束
  - 可选平滑

### 11) `emg/infer/generate_clip.py`
- `main()`：推理入口
  - 读取 ckpt + config
  - 生成 `q_clip.csv`

### 12) `emg/train/eval_basic.py`
- `compute_metrics(...)`：约束、速度、平滑度
- `main()`：输出 `report.json`

## 阅读建议（最快路径）
1) 先看：`tools/convert_csv_to_parquet.py` → 数据结构
2) 再看：`ClipDataset`/`collate_batch` → 模型输入
3) 再看：`ConditionEncoder` + `UNet1D` + `RectifiedFlow`
4) 最后看：`train_flow.py` + `generate_clip.py` + `postprocess.py`

