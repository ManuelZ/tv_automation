# Standard Library imports
from pathlib import Path
import logging
from typing import Tuple

# External imports
try:
    import pyjevois
    import libjevois as jevois
except:
    print("Not on Jevois, skipping Jevois libraries import.")
import cv2
import numpy as np

# Local imports
from config import APPS, MIN_VARIANCE_OF_LAPLACIAN, TARGET_SIZE, DEBUG, WORLD
from tv_processing import (
    center_crop,
    calc_blur,
    draw_boxes,
    get_class_from_prediction,
)
from tv_processing import (
    compute_bounding_box_coordinates,
    crop_image_patch,
    equalize_luminance,
    call_while_error,
    get_hough_lines,
    mask_image,
    identify_selected_app_v1,
    identify_selected_app_v2,
    get_boxes,
    draw_hough_lines,
)
from localization import BayesFilter


if "pyjevois" in globals():
    log_filepath = Path("/", "jevois", "data", "log.log")
else:
    log_filepath = Path("log.log")

logging.basicConfig(
    filename=log_filepath,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class DummyTimer:
    def start(self):
        return

    def stop(self):
        return


class TvAutomation:
    def __init__(self, net_path=None):
        self.on_jevois = "pyjevois" in globals()

        # Configured camera sensor resolution. Ensure it's the same as the one in videomappings.cfg
        self.h, self.w = (480, 640)

        # Resized image height passed to network
        self.target_height, self.target_width = TARGET_SIZE

        # Value scaling factor applied to input pixels
        self.scale = 1.0 / 255
        # FIXME: When I use the mean I get no identifications.
        # self.mean = [123.68, 116.77, 103.93]
        self.mean = [0, 0, 0]
        self.n = 0
        self.apps = APPS
        self.classes = list(self.apps.values())
        self.colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
        }

        self.timer = DummyTimer()
        if self.on_jevois:
            self.timer = jevois.Timer("TV Automation", 10, jevois.LOG_DEBUG)
            net_path = Path(
                pyjevois.share + "dnn", "custom", "tv_apps_detect_and_classify.onnx"
            )
        assert net_path is not None and Path(net_path).is_file()

        self.net = cv2.dnn.readNet(net_path, "")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.filter = BayesFilter(WORLD)

    def preprocess_image(self, image):
        """
        Preprocesses an image by cropping a centered square region and resizing it to a target size.

            1. Crops a square region from the center of the input image.
            2. Resizes the cropped image to the specified target size while maintaining the aspect ratio.

        Args:
            image (numpy.ndarray): The input image array with shape (height, width, channels).

        Returns:
            tuple:
                - numpy.ndarray: The cropped and resized image with shape (target_size[0], target_size[1], channels).
                - float: The x-coordinate of the top-left corner of the cropped region in the original image.
                - float: The y-coordinate of the top-left corner of the cropped region in the original image.
                - float: The ratio of the target height to the smaller dimension of the original image.
                - float: The ratio of the target width to the smaller dimension of the original image.
        """

        h_ratio, w_ratio = np.array((self.target_height, self.target_width)) / min(
            [self.h, self.w]
        )

        cropped_image, top_left_x, top_left_y = center_crop(image)

        resized_image = cv2.resize(
            cropped_image,
            dsize=(self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )
        return resized_image, top_left_x, top_left_y, h_ratio, w_ratio

    def detect_objects(self, image):
        """
        Perform detect_objectsence on the input image using a pre-trained detection model and extract bounding boxes from the
        model's outputs.

        Args:
            image (numpy.ndarray): The input image array with shape (height, width, channels).

        Returns:
            list: A list of bounding boxes predicted by the model. Each box is typically represented as a list or array
                  containing coordinates and dimensions, depending on the implementation of the `get_boxes` function.
        """

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self.scale,
            size=(self.target_height, self.target_width),
            swapRB=True,
            mean=(0, 0, 0),  # Adjust mean values if necessary
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        results: list[Tuple] = get_class_from_prediction(outputs, classes=self.classes)
        # return get_boxes(outputs)
        return results

    def draw_text(self, image, text, color="green"):
        """ """
        cv2.putText(
            image,
            text=text,
            org=(3, self.h - 6),  # Bottom-left corner of the text string in the image.
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=self.colors.get(color, "white"),
        )
        return image

    def store_image(self, image, subfolder):
        """ """
        folder = Path("/", "jevois", "data", subfolder)
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"image_{self.n:08d}.png"
        filepath = str(folder / filename)
        cv2.imwrite(filepath, img=image)

    def process(self, inframe, out_frame):
        """ """

        logger.debug("Starting detection")
        self.timer.start()

        image_original = inframe.getCvBGR() if self.on_jevois else inframe
        im_original_drawn = image_original.copy()

        # If the detected image is blurry, skip it
        if calc_blur(image_original) < MIN_VARIANCE_OF_LAPLACIAN:
            logger.debug("Blurry image")
            self.timer.stop()
            return

        preprocess_results = self.preprocess_image(image_original)
        image_resized, trans_x, trans_y, h_ratio, w_ratio = preprocess_results

        # boxes = self.detect_objects(image_resized)
        results = self.detect_objects(image_resized)

        if len(results) < 1:
            text = "No apps detected"
            im_original_drawn = self.draw_text(im_original_drawn, text, color="white")

            if self.on_jevois:
                self.store_image(image_original, subfolder="no_detected")
                out_frame.sendCv(im_original_drawn)

            self.timer.stop()
            return

        boxes, scores, selected_app = results[0]
        boxes = [boxes]

        im_original_drawn = self.draw_text(
            im_original_drawn, selected_app, color="green"
        )

        im_original_drawn = draw_boxes(
            im_original_drawn,
            boxes,
            (1 / h_ratio),
            (1 / w_ratio),
            trans_x,
            trans_y,
        )

        fps = self.timer.stop()
        logger.debug(f"Finished detection. FPS={fps} fps.")
        self.n += 1

        if self.on_jevois:
            self.store_image(image_original, subfolder=f"detected/{selected_app}")
            out_frame.sendCv(im_original_drawn)
        elif DEBUG:
            cv2.imshow(f"Detections", im_original_drawn)
            cv2.waitKey(0)


if __name__ == "__main__":
    net_path = "runs/detect/train18/weights/best.onnx"
    automation = TvAutomation(net_path)

    TARGET_DIR = Path("data", "originals")
    for filename in TARGET_DIR.rglob("*.png"):
        # cv2.destroyAllWindows()
        print(f"Analyzing file {filename}")
        image_original = cv2.imread(str(filename))
        automation.process(image_original, None)
