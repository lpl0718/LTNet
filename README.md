[简体中文](README.zh-CN.md) | **English**
# [LTNet: Lightweight Tiny-Defect Detection for Intelligent PCB Manufacturing](LTNet.md)

## 📌 Important Usage Notes
**This model is also suitable for other small-to-medium scale, unobstructed tiny object detection scenarios.**

1. **Module Name Correspondence**
The **MILABlock** mentioned in the paper (renamed by the original author) is actually implemented as **MLLABlock** in the code.

2. **Input Requirements**
**This module only supports square feature map inputs**, please ensure:
- ⚠️ **Validation set and test set images must be square. If necessary, achieve this by padding with black borders**
- ✅ **Due to YOLO's training mechanism, there are no mandatory size requirements for the training set**

3. **Custom Dataset Adaptation**
- Model configurations are located in the folder:  
  `LTNet_cfg`
- Modify the regression loss function to adapt to your dataset:  
  `ultralytics/utils/loss.py` **Line 170**
- Customize optimal model selection logic:  
  `ultralytics/utils/metrics.py` **Line 970**

## 📌 Environment Setup (Optional)
✅**No special environment requirements**  
**The following steps can partially reproduce my environment ↓↓↓**
```bash
conda create -n ltnet python=3.10
conda activate ltnet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```  
**My environment is compatible with numpy==2.2.6, but when I re-create my environment, package conflicts occur, so I lowered the version to numpy<2 to avoid conflicts**  
**I have packaged my environment at ↓↓↓↓↓↓**  
`https://pan.quark.cn/s/859f5afe76bc`
  

## 📌Datasets
**Dataset 1: PKU-Market-PCB provided by Intelligent Robot Open Laboratory of Peking University**  
**Dataset 2: DeepPCB provided by Shanghai Jiao Tong University Institute of Image Processing and Pattern Recognition**  
**Specific dataset splits and augmentation strategies ↓↓↓**  
`https://pan.quark.cn/s/4929ce1e571b`

**Dataset 3: DsPCBSD+**  
`https://figshare.com/articles/dataset/DsPCBSD_/24970329?file=44069552`

> **⚠️ Important Note**
> When training with the DeepPCB dataset, please ensure to use `scale=0`, which is the direct cause of poor training results for most models.
