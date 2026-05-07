**简体中文** | [English](README.md)
# [LTNet: 一种轻量且高精度的 PCB 表面缺陷检测网络](LTNet.md)

## 📌 重要使用说明
**本模型同样适用于其他中小规模、无遮挡的极小目标检测场景。**

1. 模块名称说明  
论文中的 **MILABlock**（原作者在论文中已更名）在代码中实际命名为 **MLLABlock**。

2. 输入要求
**该模块仅支持方形特征图输入**，请务必保证：
- ⚠️ **验证集、测试集图片必须为正方形,如果有必要建议以填充黑边的方式实现**
- ✅ **受YOLO训练机制影响，训练集无强制尺寸要求**

3. 自定义适配数据集
- 模型配置在文件夹：  
  `LTNet_cfg`
- 修改回归损失函数适配你的数据集：  
  `ultralytics/utils/loss.py` **Line  170**
- 自定义最优模型筛选逻辑：  
  `ultralytics/utils/metrics.py` **Line  970**


## 📌环境配置（不是必要的）
✅**没有特殊的环境要求**  
**以下方式可以部分还原我的环境↓↓↓**
```bash
conda create -n ltnet python=3.10
conda activate ltnet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```  

**我的环境兼容numpy==2.2.6，但当我重新复现我的环境时会发生安装包冲突，为避免冲突降低版本numpy<2**  
**我把我的环境打包在↓↓↓**  
`https://pan.quark.cn/s/859f5afe76bc`
  

## 📌数据集
**数据集1：PKU-Market-PCB 北京大学智能机器人开放实验室提供**  
**数据集2：DeepPCB 上海交通大学图像处理与模式识别研究所提供**  

**具体数据集划分与增强策略↓↓↓**  
`https://pan.quark.cn/s/4929ce1e571b`







