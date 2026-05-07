from ultralytics import YOLO




def main():


    model = YOLO(r'')


    model.train(data=r"",
                imgsz=640,workers=4,epochs=180,batch=32,patience=60,
                plots=True, exist_ok=False,
                close_mosaic=20,amp=True,freeze=0,optimizer='SGD',save_period=10,lrf=0.005,
                hsv_h =0.0 ,hsv_s =0.0 ,hsv_v= 0.0,degrees= 0.0, # Disable these data augmentations.
                scale=0.0       # Due to dataset limitations, please be sure to disable this augmentation. This is also why other specialized models perform poorly.
                )


if __name__ == '__main__':
    main()