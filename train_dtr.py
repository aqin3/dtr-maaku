from ultralytics import YOLO

# paths
DATA_YAML = "../datasets/640/data.yaml"  
MODEL = "yolov8n.pt"                  # try yolov8s.pt as well

# recommended overrides
overrides = dict(
    data=DATA_YAML,
    imgsz=640,
    epochs=100,
    batch=64,            # 32 if VRAM limited; 64 is fine on 4080
    optimizer="AdamW",
    lr0=0.003,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=25,         # early stop
    hsv_h=0.005, hsv_s=0.4, hsv_v=0.35,
    degrees=5.0, translate=0.05, scale=0.2, shear=2.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0,          # enable but we’ll turn it off late
    mixup=0.0,
    erasing=0.1,         # coarse dropout
    close_mosaic=30,     # disable mosaic for last 30 epochs
    ema=True,
    imgsz_val=640,       # keep eval size same during training
)

# train n model
model_n = YOLO(MODEL)
model_n.train(**overrides)

# optional: train s model for comparison
# overrides["batch"] = 32  # s uses more VRAM
# YOLO("yolov8s.pt").train(**overrides)

