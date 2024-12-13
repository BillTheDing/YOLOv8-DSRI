import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8-main/runs/train/C2f-DWR-RFA3/weights/best.pt') # select your model.pt path
    model.predict(source='/root/autodl-tmp/yolov8-main/newdata3/test/IMG_20220927_134813.jpg',
                project='runs/detect',
                name='exp',
                save=True,max_det=1200,imgsz=1280,hide_labels=True,conf=0.3,
                # visualize=True # visualize model features maps
                )