# SL-Former: 基于空间语义编码与Transformer的结构地震时程响应实时预测

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenSeesPy](https://img.shields.io/badge/OpenSeesPy-3.5+-green.svg)](https://openseespydoc.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-blueviolet.svg)](https://arxiv.org/)


## 📖 项目简介

**SL-Former** (Spatial-Language Transformer) 是一个基于Transformer架构的结构地震时程响应实时预测系统。它通过创新的**空间语义编码**策略，将建筑结构的空间布置信息与地震动时程进行深度融合，实现了对结构顶点位移时程的快速、准确预测。

### 🎯 核心创新

| 创新点 | 说明 |
|--------|------|
| **空间语义编码** | 摒弃传统刚度矩阵，通过参数化/八叉树编码将建筑空间布置压缩为低维特征向量 |
| **双路交叉注意力** | 实现"荷载激励查询结构刚度"与"结构薄弱区查询敏感地震时刻"的双向物理耦合 |
| **物理知情正则化** | 基于经验周期约束预测位移频谱，保证物理合理性 |
| **多进程并行仿真** | 集成OpenSeesPy实现大规模数据生成，充分利用多核CPU资源 |

### 📊 性能指标

| 指标 | 传统求解器 (Newmark-β) | SL-Former |
|------|----------------------|-----------|
| 单次预测耗时 | ~22秒 | **~0.016秒** |
| 峰值位移误差 | - | **< 8%** |
| 波形相似度 (NSE) | - | **> 0.90** |
| R² | - | **0.952** |
| 计算加速比 | 1× | **> 1400×** |

## 🏗️ 系统架构

[1] 数据生成层

│

├── 结构参数生成 (8维: 层数/跨数/跨宽/层高/质量/阻尼)

├── 地震动加载 (天然波 + 合成波, 80条)

├── 3D OpenSees仿真 (多进程并行, 16进程)

└── 仿真结果缓存 (位移时程 + 结构参数)

│

▼

[2] 特征提取层

│

├── 直接参数编码 (8维 → 128维)

├── 八叉树编码 (体素→96维特征, 可选)

└── 统计特征编码 (体积分数/质心偏移, 可选)

│

▼

[3] Transformer层

│

├── 时序编码 (加速度 → 位置编码)

├── 结构特征注入 (融合到每个时间步)

├── 双路交叉注意力 (4层)

│ ├── 路径A: 时序 → 结构 (荷载查询刚度)

│ └── 路径B: 结构 → 时序 (结构查询敏感时刻)

└── 前馈网络 + 残差连接

│

▼

[4] 输出层

│

├── 顶点位移时程预测 [T]

├── 物理正则化约束 (频谱约束)

└── 反归一化输出


### 数据流详解

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| **仿真生成** | 结构参数 [8] + 地震动 [T] | OpenSees 3D时程分析 | 位移时程 [T] |
| **特征提取** | 结构参数 [8] | MLP编码 | 结构特征 [128] |
| **时序编码** | 加速度时程 [T,1] | 线性投影 + 位置编码 | 时序特征 [T,256] |
| **交叉注意力** | 时序特征 + 结构特征 | 4层双路交叉注意力 | 融合特征 [T,256] |
| **输出** | 融合特征 | MLP解码 | 位移时程 [T] |

## 📁 目录结构

SL-Former/

├── config.py # 全局配置

├── dataset.py # 数据集与八叉树特征生成

├── transformer_model.py # Transformer模型定义

├── train.py # 训练脚本

├── evaluate.py # 评估与可视化

├── simulation_cache.py # 仿真缓存管理（多进程并行）

├── earthquake_simulator_3d.py # 3D OpenSees仿真器

├── octree_encoder.py # 八叉树编码器

├── voxel_converter.py # 体素矩阵转换

├── generate_frames.py # 框架结构生成

├── physics_loss.py # 物理正则化损失

├── run_pipeline.py # 一键运行全流程

├── models/ # 训练好的模型

├── cache/ # 缓存目录

│ ├── simulation_cache.pkl # 仿真结果缓存

│ └── octree_cache.pkl # 八叉树特征缓存

└── plots/ # 可视化图表输出


## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，GPU加速)
- 16GB+ RAM (推荐128GB用于大规模数据生成)

### 安装依赖

# 克隆仓库
git clone https://github.com/yourusername/SL-Former.git
cd SL-Former

# 创建虚拟环境
conda create -n slformer python=3.10
conda activate slformer

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openseespy numpy scipy matplotlib scikit-learn tqdm tensorboard


## 运行完整流程

### 1. 生成5000个仿真样本（自动使用多进程并行）
python run_pipeline.py --force_sim --num_samples 5000

### 2. 生成八叉树特征
python run_pipeline.py --force_octree --skip_train

### 3. 训练模型
python train.py

### 4. 评估模型并生成可视化图表
python evaluate.py

## 各模块独立运行

### 仅生成仿真数据
python simulation_cache.py

### 仅训练（使用已有缓存）
python train.py

### 仅评估
python evaluate.py


## 命令行参数

###参数	说明

--force_sim	强制重新生成仿真数据

--force_octree	强制重新生成八叉树特征

--skip_train	跳过训练（仅生成数据）

--num_samples N	指定仿真样本数

## 📊 结果可视化

运行评估后，./plots/ 目录下会生成以下图表：

图表文件	内容描述
structure_layout.png	结构平面图 + 立面图
evaluation_overview.png	预测vs真实散点图、误差分布、峰值误差
error_analysis.png	6子图误差分析 (误差vs真实值、Q-Q图、RMSE分布等)
peak_scatter.png	峰值预测散点图 (按高度着色)
height_analysis.png	按高度分组的性能对比
sample_comparison.png	加速度 + 位移 + 误差 三列时程对比

## 示例输出

### 预测 vs 真实位移时程
https://plots/sample_comparison.png

### 评估总览
https://plots/evaluation_overview.png

## 🔬 理论方法

### 1. 空间语义编码

传统方法依赖质量/刚度矩阵（M/C/K），需要完整的有限元建模。SL-Former提出空间语义编码策略：

直接参数编码：使用8维结构参数（层数、跨数、跨宽、层高、质量、阻尼比）

八叉树编码：将 [300,300,500] 体素矩阵压缩为96维特征（可选扩展）

统计特征编码：提取体积分数、质心偏移等结构统计特征

### 2. 双路交叉注意力

路径A: 时序 → 结构  (Q=加速度, K/V=结构特征)
  物理含义: 当前地震动分量主要激发哪些区域
路径B: 结构 → 时序  (Q=结构特征, K/V=加速度)
  物理含义: 结构薄弱区对地震动中哪些时刻最敏感

### 3. 物理知情正则化
基于经验周期估算约束预测位移频谱

## 📝 配置说明

所有参数集中在 config.py 中管理：

## 🖥️ 硬件要求
组件	最低配置	推荐配置
CPU	8核心	16核心+ (并行仿真)
内存	16GB	128GB
GPU	8GB显存	16GB+ 显存
存储	50GB	100GB SSD

## 📚 依赖项

torch >= 2.0.0
openseespy >= 3.5.0
numpy >= 1.24.0
scipy >= 1.10.0
matplotlib >= 3.7.0
scikit-learn >= 1.2.0
tqdm >= 4.65.0
tensorboard >= 2.13.0

## 📧 联系方式
如有任何问题，欢迎联系项目作者。

## 📄 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件
