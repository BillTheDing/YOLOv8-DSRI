from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
model = YOLO("test/yolov8.pt")  # 模型文件路径
results = model("test/img/IMG_20220927_134813.jpg", visualize=True)  # 要预测图片路径和使用可视化
