# Standard Library imports
import os
from enum import Enum
from pathlib import Path

# External imports
from ultralytics import YOLO


class Mode(Enum):
    TRAIN_DETECTION = 1
    TRAIN_CLASSIFICATION = 2


def load_model(mode):
    """ """

    # COCO-pretrained
    if mode == Mode.TRAIN_DETECTION:
        model_path = os.path.join(".", "models", "yolov8n.pt")

    # Pretrained
    elif mode == Mode.TRAIN_CLASSIFICATION:
        model_path = os.path.join(".", "models", "yolov8n-cls.pt")

    return model_path


if __name__ == "__main__":
    mode = Mode.TRAIN_DETECTION
    model_path = load_model(mode)
    model = YOLO(model_path)

    if mode == Mode.TRAIN_DETECTION:
        # imgsz must be a multiple of 32
        results = model.train(data="config.yml", epochs=100, imgsz=256, workers=0)

        # https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
        model.export(format="onnx", half=True)

    if mode == Mode.TRAIN_CLASSIFICATION:
        data_dir = Path("data", "classification_yolov8", "images")

        # imgsz must be a multiple of 32
        results = model.train(
            data=data_dir, epochs=100, imgsz=256, workers=0, augment=False
        )

        # https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
        model.export(format="onnx", half=True)
