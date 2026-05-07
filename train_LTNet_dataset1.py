from ultralytics import YOLO




def main():



    model = YOLO(r'LTNet_cfg/LTNet_L.yaml')

    model.train(data=r"C:\Users\13780\Desktop\PkPCB_A\PkPCB_A.yaml",imgsz=640,workers=8,epochs=140,batch=8,patience=60,
                plots=True,exist_ok=False,
                close_mosaic=0,amp=True,freeze=0,
                optimizer='SGD',rect=False,
                 lrf=0.005,save_period=10,
                 hsv_h =0.0 ,hsv_s =0.0 ,hsv_v=0.0   # Disable these data augmentations.
                )


if __name__ == '__main__':
    main()