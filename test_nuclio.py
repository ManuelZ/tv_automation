# Standard Library imports
from pathlib import Path
import base64
import io

# External imports
import cv2
import numpy as np
from PIL import Image
import requests


# Note: Replace the port. Run `docker container ps` and look for the port being used
# by your Nuclio function container
ENDPOINT_URL = "http://localhost:57062"


def generate_dummy_image(width=100, height=100, color=(255, 0, 0)):
    """Generate a dummy image with a single color."""
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image_array[:, :] = color
    image = Image.fromarray(image_array)
    return image


# Convert PIL image to Base64
def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def pil_to_numpy(im):
    im_numpy = np.array(im)
    im_numpy = cv2.cvtColor(im_numpy, cv2.COLOR_BGR2RGB)
    return im_numpy


# Send the image to the endpoint
def send_image_to_endpoint(image, endpoint_url, threshold=0.5):
    image_data = pil_to_base64(image)
    payload = {"image": image_data, "threshold": threshold}
    response = requests.post(endpoint_url, json=payload)
    if response.status_code == 200:
        print("Image sent successfully.")
        print("Response:", response.json())
        return response.json()
    else:
        print("Failed to send image.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


im_path = Path("images/capture_example.png")
image = Image.open(im_path)
im_numpy = pil_to_numpy(image)

# Example usage
response = send_image_to_endpoint(image, ENDPOINT_URL)

for obj in response:
    confidence = float(obj["confidence"])
    label = obj["label"]
    points = obj["points"]

    # Convert float points to integer
    x1, y1, x2, y2 = map(int, points)

    # Draw rectangle
    cv2.rectangle(im_numpy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add label and confidence
    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        im_numpy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    cv2.imshow("results", im_numpy)
    cv2.waitKey(0)
