# Standard Library imports
import json
import base64
import io

# External imports
import numpy as np
import cv2
from ultralytics import YOLO


def init_context(context):
    context.logger.info("Init context...  0%")

    classes = [
        "youtube",
        "television",
        "netflix",
        "max",
        "internet",
        "prime_video",
        "live_tv",
        "movistar_tv_app",
        "spotify",
    ]

    model = YOLO("/opt/nuclio/best.pt")  # pretrained YOLOv8n model

    context.user_data.model_handler = model
    context.user_data.classes = classes

    context.logger.info("Init context...100%")


def get_yolov8_results_for_cvat(predictions, classes, conf_threshold=0.4):
    """ """
    results = []
    for pred in predictions:
        boxes = pred.boxes
        names = pred.names

        if boxes is None or names is None:
            continue

        for box in boxes:
            print(f"Predicted box", box)
            conf = box.conf
            label = classes[int(box.cls)]
            points = box.xyxy.tolist()[0]

            if conf >= conf_threshold:
                results.append(
                    {
                        "confidence": str(float(conf)),
                        "label": str(label),
                        "points": points,
                        "type": "rectangle",
                    }
                )
    print(results)
    return results


def bytes_to_numpy(file_bytes: bytes) -> np.ndarray:
    """ """
    numpy_array = np.frombuffer(file_bytes, np.uint8)
    image_numpy = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image_numpy


def handler(context, event):
    """ """
    context.logger.info("Run Yolov8 nano model")

    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = bytes_to_numpy(buf.getvalue())

    predictions = context.user_data.model_handler(image)
    results = get_yolov8_results_for_cvat(predictions, context.user_data.classes)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
