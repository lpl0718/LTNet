# ultralytics/utils/yolo_size_stats.py
import numpy as np
import torch


class YOLOSizeStats:
    """专门针对YOLO格式数据集的尺寸统计器."""

    def __init__(self, img_size=640):
        self.img_size = img_size
        self.categories = ["tiny", "small", "medium", "large"]
        self.reset()

    def reset(self):
        self.stats = {}
        for cat in self.categories:
            self.stats[cat] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        self.total = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}

    def yolo_to_xyxy(self, yolo_bbox):
        """将YOLO格式的xywh转换为xyxy格式."""
        if len(yolo_bbox) < 4:
            return None

        x_center, y_center, width, height = yolo_bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # 确保坐标在[0,1]范围内
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))

        return [x1, y1, x2, y2]

    def get_size(self, bbox, is_yolo_format=True):
        """获取目标尺寸分类.

        Args:
            bbox: 边界框坐标 [x1,y1,x2,y2] 或 [x_center,y_center,width,height]
            is_yolo_format: 是否为YOLO格式的xywh
        """
        if len(bbox) < 4:
            return "unknown"

        # 如果是YOLO格式，转换为xyxy
        if is_yolo_format:
            xyxy_bbox = self.yolo_to_xyxy(bbox)
            if xyxy_bbox is None:
                return "unknown"
            x1, y1, x2, y2 = xyxy_bbox
        else:
            # 假设已经是xyxy格式
            x1, y1, x2, y2 = bbox

        # 转换为像素尺寸
        w_pixel = (x2 - x1) * self.img_size
        h_pixel = (y2 - y1) * self.img_size
        area = w_pixel * h_pixel

        if area < 16 * 16:
            return "tiny"
        elif area < 32 * 32:
            return "small"
        elif area < 64 * 64:
            return "medium"
        else:
            return "large"

    def box_iou(self, box1_xyxy, box2_yolo):
        """计算IoU.

        Args:
            box1_xyxy: 预测框，xyxy格式（归一化）
            box2_yolo: 真实框，YOLO格式（归一化xywh）
        """
        # 确保是numpy数组
        if isinstance(box1_xyxy, torch.Tensor):
            box1_xyxy = box1_xyxy.cpu().numpy()
        if isinstance(box2_yolo, torch.Tensor):
            box2_yolo = box2_yolo.cpu().numpy()

        # 检查无效值
        if np.any(np.isnan(box1_xyxy)) or np.any(np.isnan(box2_yolo)):
            return 0.0

        # 将YOLO格式转换为xyxy
        box2_xyxy = self.yolo_to_xyxy(box2_yolo)
        if box2_xyxy is None:
            return 0.0

        # 确保坐标在有效范围内
        box1_xyxy = np.clip(box1_xyxy, 0, 1)
        box2_xyxy = np.clip(box2_xyxy, 0, 1)

        # 计算IoU
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])

        union = area1 + area2 - inter
        if union <= 0:
            return 0.0

        return inter / union

    def update(self, preds, gt_bboxes, gt_labels, img_id=0, iou_thres=0.5):
        """更新统计 - 简化的接口.

        Args:
            preds: 预测结果 [N, 6] (xyxy归一化, conf, cls)
            gt_bboxes: 真实边界框 [M, 4] (YOLO格式: x_center, y_center, width, height)
            gt_labels: 真实标签 [M] (class_id)
            img_id: 图像ID（在batch中的索引）
            iou_thres: IoU阈值
        """
        if len(gt_bboxes) == 0:
            # 没有真实目标，所有预测都是FP
            if preds is not None and len(preds) > 0:
                for pred in preds:
                    pred_box = pred[:4].cpu().numpy()
                    size_cat = self.get_size(pred_box, is_yolo_format=False)
                    if size_cat in self.stats:
                        self.stats[size_cat]["pred"] += 1
                        self.stats[size_cat]["fp"] += 1
                        self.total["pred"] += 1
                        self.total["fp"] += 1
            return

        # 统计真实目标
        for i, (gt_label, gt_bbox) in enumerate(zip(gt_labels, gt_bboxes)):
            gt_bbox_np = gt_bbox.cpu().numpy()
            size_cat = self.get_size(gt_bbox_np, is_yolo_format=True)
            if size_cat in self.stats:
                self.stats[size_cat]["gt"] += 1
                self.total["gt"] += 1

        # 如果没有预测，所有真实目标都是FN
        if preds is None or len(preds) == 0:
            for i, (gt_label, gt_bbox) in enumerate(zip(gt_labels, gt_bboxes)):
                gt_bbox_np = gt_bbox.cpu().numpy()
                size_cat = self.get_size(gt_bbox_np, is_yolo_format=True)
                if size_cat in self.stats:
                    self.stats[size_cat]["fn"] += 1
                    self.total["fn"] += 1
            return

        # 处理预测
        pred_boxes = preds[:, :4]
        pred_scores = preds[:, 4]
        pred_cls = preds[:, 5]

        # 按置信度排序
        if len(pred_scores) > 0:
            sorted_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_idx]
            pred_cls = pred_cls[sorted_idx]
            pred_scores = pred_scores[sorted_idx]

        gt_matched = [False] * len(gt_bboxes)

        # 匹配预测和真实目标
        for i, (pred_box, pred_score, pred_cls_idx) in enumerate(zip(pred_boxes, pred_scores, pred_cls)):
            # 获取预测框尺寸类别
            pred_box_np = pred_box.cpu().numpy()
            pred_size_cat = self.get_size(pred_box_np, is_yolo_format=False)

            if pred_size_cat not in self.stats:
                continue

            self.stats[pred_size_cat]["pred"] += 1
            self.total["pred"] += 1

            best_iou = 0
            best_idx = -1

            # 寻找最佳匹配的真实目标
            for j, (gt_label, gt_bbox) in enumerate(zip(gt_labels, gt_bboxes)):
                if gt_matched[j]:
                    continue

                # 类别必须匹配
                if int(gt_label) != int(pred_cls_idx):
                    continue

                gt_bbox_np = gt_bbox.cpu().numpy()
                iou = self.box_iou(pred_box_np, gt_bbox_np)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou > iou_thres and best_idx >= 0:
                # TP：匹配成功
                gt_bbox_np = gt_bboxes[best_idx].cpu().numpy()
                gt_size_cat = self.get_size(gt_bbox_np, is_yolo_format=True)

                if gt_size_cat == pred_size_cat:
                    self.stats[pred_size_cat]["tp"] += 1
                    self.total["tp"] += 1
                    gt_matched[best_idx] = True
                else:
                    # 尺寸类别不匹配，视为FP
                    self.stats[pred_size_cat]["fp"] += 1
                    self.total["fp"] += 1
            else:
                # FP：未匹配到真实目标
                self.stats[pred_size_cat]["fp"] += 1
                self.total["fp"] += 1

        # 统计FN（未匹配的真实目标）
        for j, matched in enumerate(gt_matched):
            if not matched:
                gt_bbox_np = gt_bboxes[j].cpu().numpy()
                size_cat = self.get_size(gt_bbox_np, is_yolo_format=True)
                if size_cat in self.stats:
                    self.stats[size_cat]["fn"] += 1
                    self.total["fn"] += 1

    def get_metrics(self):
        """计算并返回所有指标."""
        metrics = {}

        for cat in self.categories:
            s = self.stats[cat]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]

            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            ap = precision * recall  # 简化的AP计算

            detection_rate = s["pred"] / max(s["gt"], 1)

            metrics[cat] = {
                "gt": s["gt"],
                "pred": s["pred"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "precision": float(precision),
                "recall": float(recall),
                "ap": float(ap),
                "detection_rate": float(detection_rate),
            }

        # 总体指标
        total_tp, total_fp, total_fn = self.total["tp"], self.total["fp"], self.total["fn"]
        total_precision = total_tp / (total_tp + total_fp + 1e-16)
        total_recall = total_tp / (total_tp + total_fn + 1e-16)
        total_detection_rate = self.total["pred"] / max(self.total["gt"], 1)

        metrics["total"] = {
            "gt": self.total["gt"],
            "pred": self.total["pred"],
            "tp": self.total["tp"],
            "fp": self.total["fp"],
            "fn": self.total["fn"],
            "precision": float(total_precision),
            "recall": float(total_recall),
            "detection_rate": float(total_detection_rate),
        }

        return metrics
