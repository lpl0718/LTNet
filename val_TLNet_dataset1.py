from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model=r"LTNet_result/PKU-Market-PCB/M/best.pt")

    metrics = model.val(
        data=r"Quick_test.yaml",
        split="test",
        imgsz=640,
        batch=32,
        save_json=False,
        plots=True,
        verbose=True,
        visualize=True,
        save_conf=True,
        amp=True,
        exist_ok=True,
        workers=8,
        max_det=300,
        augment=False,
    )
