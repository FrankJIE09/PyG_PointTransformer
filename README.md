# PyG PointTransformer 点云分割模型 (基于 ShapeNetPart)

本项目基于 [PyTorch Geometric (PyG)] 构建了一个 PointTransformer 模型，用于对点云数据进行分割任务。数据采用 ShapeNetPart HDF5 格式。

---

## 📁 项目结构

```
.
├── dataset.py        # 加载并增强 ShapeNetPart 数据集
├── model.py          # 基于 PyG 的 PointTransformer 点云分割模型
├── train.py          # 训练与验证主程序
├── test.py           # 检查 HDF5 数据文件结构
```

---

## 📦 环境依赖

建议使用 Conda 环境，依赖主要包括：

- Python 3.12+
- PyTorch
- PyTorch Geometric
- h5py, numpy, tqdm 等

详见下方 [`environment.yml`](#📋-conda环境配置)。

---

## 📊 数据准备

请将 ShapeNetPart HDF5 数据放置于指定目录，例如：

```
./data/shapenetpart_hdf5_2048/train*.h5
./data/shapenetpart_hdf5_2048/test*.h5
```

推荐数据集来源：[PointNet Data](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip)

---

## 🚀 训练与验证

```bash
# 使用默认参数运行训练
python _train.py --data_root ./data/shapenetpart_hdf5_2048
```

支持的关键参数包括：

```bash
--num_points 2048            # 每个点云采样点数
--batch_size 48              # 批大小
--epochs 100                 # 训练轮数
--num_classes 50             # 类别数量（ShapeNetPart 为 50）
--checkpoint_dir ./checkpoints
--resume                     # 从上一次 best_model 恢复训练
```

---

## 🧪 测试数据文件结构

你可以使用 `test.py` 脚本检查你的 HDF5 文件结构：

```bash
python test.py
```

确保 `data` 和 `seg` 键存在于 HDF5 文件中。

---

## 🧠 模型结构

模型主要由以下模块构成：

- 特征嵌入 MLP
- 多层 PointTransformerConv（PyG 实现）
- 分割头（解码器）

---

## 💾 Checkpoints

训练过程中会保存最佳模型：

```
./checkpoints_seg_pyg_ptconv/best_model.pth
```

你可以通过 `--resume` 参数恢复训练。

---

## 📋 Conda环境配置

详见下方 `environment.yml`。
```
name: pyg_pointtransformer_seg
channels:
  - pytorch
  - pyg
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pytorch>=1.12
  - pyg>=2.3.0
  - torchvision
  - torchaudio
  - h5py
  - numpy
  - tqdm
  - scikit-learn
  - pip
  - pip:
      - torch-geometric
```