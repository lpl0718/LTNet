> 🔙 [Return to the main README](README.md)
# LTNet: Lightweight Tiny-Defect Detection for Intelligent PCB Manufacturing(Journal of Intelligent Manufacturing，Pending Online)
## Abstract
Quality control of Printed Circuit Boards (PCBs) is critical for ensuring their performance and reliability. Although deep learning-based methods have been widely adopted for surface defect inspection in PCB manufacturing, several challenges remain: sparse low-saliency defect features, complex background interference, class imbalance, and the high computational cost of existing models. To mitigate these issues, this work proposes LTNet (Lightweight Tiny-Object Network), a lightweight detector built upon YOLOv11n. LTNet incorporates four key components: (1) a C3K2-MILABlock for enhanced feature representation, (2) an LT-Backbone derived from a systematic analysis of space-to-depth convolutions and $2{\times}2$ strided convolutions, (3) a lightweight multi-scale neck with an Adaptive Feature Compression Module (AFCM), and (4) a Normalized Gaussian Wasserstein Distance (NWD) loss for stable small-box regression. Three variants (S, M, and L) accommodate different resource budgets. Across five random seeds, LTNet-L achieves 58.28\% mAP, 98.58\% mAP$_{50}$, 96.88\% precision, and 97.44\% recall on PKU-Market-PCB with only 1.02\,M parameters and 12.7\,GFLOPs at $640{\times}640$ resolution. Evaluations on DeepPCB and DsPCBSD+ further show consistent improvements over the YOLOv11n baseline, particularly for ultra-tiny and tiny defects, while achieving competitive performance against state-of-the-art PCB-specific defect detectors. An RK3588 (Rockchip) deployment test demonstrates the feasibility of running LTNet-M at approximately 5\,FPS, indicating its viability as a lightweight inspection component for PCB quality-control workflows. The source code is available at \url{https://github.com/lpl0718/LTNet}.
## Keywords
Deep learning, PCB defect detection, Manufacturing quality inspection, Lightweight network, Attention mechanism

## Performance Comparison on PKU-Market-PCB and DeepPCB Datasets

| Model | Res. | P(%) | R(%) | mAP50(%) | mAP(%) | Res. | P(%) | R(%) | mAP50(%) | mAP(%) | Para.(M) | GFLOPs | Infe.(ms) |
| --- | --- |------|------|----------|--------| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | 640 | 97   | 91.4 | 95.9     | 53     | 640 | 97.6 | 97.6 | 99 | 81 | 3 | 8.1 | 4.4 |
| YOLOv11n | 640 | 97.4 | 90.4 | 95.5     | 52.1   | 640 | 97.1 | 96.7 | 98.7 | 81.9 | 2.58 | 6.3 | 4.6 |
| YOLOv11s | 640 | 97.4 | 92.7 | 97.3     | 53.9   | 640 | 98.8 | 97.3 | 99 | 82.6 | 9.42 | 21.3 | 5.6 |
| YOLOv11m | 640 | 96.3 | 95.9 | 97.5     | 54.5   | 640 | 98 | 98.5 | 99.2 | 83.5 | 20.03 | 67.7 | 8.8 |
| RT-DETR | 640 | 95.5 | 91.9 | 94.1     | 42.2   | — | — | — | — | — | 32 | 103.5 | 11.3 |
| GoldYOLO | 640 | 96   | 93.5 | 95.3     | 52.3   | — | — | — | — | — | 9.9 | 20.5 | 6.6 |
| YOLO-RRL | 640 | 96.2 | —    | 95.2     | —      | — | — | — | — | — | 2.7 | 7.4 | — |
| MFAD-RTDETR | 640 | 96.5 | 94.5 | 97       | 51     | — | — | — | — | — | 16.2 | — | — |
| CDI-YOLO | 640 | 97.1 | 96.4 | 98.3     | 51.1   | — | — | — | — | — | 5.76 | 12.6 | — |
| Y-GhostNet | 640 | 97.9 | 94.5 | 96.7     | —      | — | — | — | — | — | 3.69 | 8.2 | — |
| Model1 | 608 | 98   | 97.9 | 97.9     | —      | — | — | — | — | — | 32.8 | — | — |
| TDD-YOLO | 640 | —    | 95   | 97.8     | 49.4   | — | — | — | — | — | — | — | — |
| RDTor | — | 97.2 | 98.8 | 99.3     | —      | — | — | — | — | — | — | — | — |
| ABF-YOLO | 640 | 95.8 | 95.8 | 97.7     | —      | — | — | — | — | — | — | — | — |
| YOLO-DFA | 640 | 96.3 | 95.9 | 96.4     | —      | — | — | — | — | — | 37.3 | 56.7 | — |
| SGT-YOLO | 640 | 98   | 97.6 | 98.6     | 58.8   | — | — | — | — | — | 1.49 | 10.2 | 4.9 |
| SDDF-Net | — | —    | —    | —        | —      | 640 | 98.8 | 98.8 | 98.3 | — | 10 | 23 | — |
| MS-YOLO | — | —    | —    | —        | —      | 640 | 96.4 | 91.7 | 96.4 | 68.5 | 4.7 | — | — |
| YOLOv8-DEE | — | —    | ---  | ---      | ---    | 640 | — | — | 98.7 | 72.9 | — | — | — |
| LPCB-YOLO | — | ---  | ---  | ---      | ---    | 640 | 93.6 | 91.1 | 95.1 | — | — | — | — |
| YOLO-EMAC | 640 | 94.6 | 98.4 | 97.3     | —      | 640 | — | — | 98.3 | — | 12.9 | 27.7 | — |
| EL-PCBNet | 640 | —    | —    | 97.5     | —      | 640 | — | — | 98.5 | — | 5.5 | 12.6 | — |
| EffNet-PCB | 640 | 96.2 | 96.1 | 95.6     | 50.3   | 640 | 98.6 | 97 | 99 | 78.6 | 6.4 | 22.2 | — |
| PSDDNet | 1280 | 97.7 | 98   | 97.7     | 57.2   | 640 | — | 94.8 | 98.3 | — | 0.932 | 55/13.8 | — |
| OURS-S | 640 | 97.6 | 95.9 | 98.1     | 56.9   | 640 | 97.8 | 95.8 | 98.9 | 81.8 | 0.53 | 5.7 | 5.9 |
| OURS-M | 640 | 97.8 | 97.7 | 98.4     | 57.8   | 640 | 97.7 | 97.4 | 99.1 | 83.4 | 0.81 | 8.5 | 6.6 |
| OURS-L | 640 | 97.5 | 98.8 | 98.8     | 58.4   | 640 | 97.3 | 97.5 | 99.1 | 84.2 | 1.02 | 12.7 | 7.7 |

> Note: Res, Para, and Infe are abbreviations of Resolution, Parameter, and Inference, respectively.
>   
  

![Visualization](Compare-1.pdf)  
   
![Visualization](Compare-deep.pdf)

![Visualization](Compare-ds.pdf)