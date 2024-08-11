FROM ultralytics/ultralytics:8.2.74-cpu

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/nuclio

# Copy model to the image
COPY runs/detect/train18/weights/best.pt best.pt
