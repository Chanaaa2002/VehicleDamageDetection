from ultralytics import YOLO

DATA_DIR = r"data/processed/archive1"   
MODEL = "yolo11n-cls.pt"

model = YOLO(MODEL)
model.train(
    data=DATA_DIR,
    imgsz=224,
    epochs=10,
    batch=32,
    name="archive1_yolo11n_cls"
)
