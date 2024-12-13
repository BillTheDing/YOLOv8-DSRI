import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8-main/ultralytics/cfg/models/v8/yolov8-C2f-DWR.yaml')
    model.load('/root/autodl-tmp/yolov8-main/yolov8n-seg.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/yolov8-main/silkworm/data.yaml',
                cache=False,
                project='runs/train',
                name='silkworm',
                epochs=100,
                batch=16,
                imgsz=640,
               # resume=True,
                close_mosaic=10,
                optimizer='SGD', # using SGD
                max_det=1200,
                workers=15,
                resume='/root/autodl-tmp/yolov8-main/runs/train/silkworm2/weights/last.pt', # last.pt path
                # amp=False # close amp
                # fraction=0.2
                )