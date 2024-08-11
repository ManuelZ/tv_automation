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
    get_prediction_results,
    counter,
)
from tv_processing import (
    calc_bbox_tlbr_coords,
    crop_image_patch,
    equalize_light,
    call_while_error,
    get_hough_lines,
    identify_selected_app_v1,
    identify_selected_app_v2,
    draw_hough_lines,
)
from localization import BayesFilter


if "pyjevois" in globals():
    log_filepath = Path("/", "jevois", "data", "log.log")
else:
    log_filepath = Path("log.log")

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG if DEBUG else logging.info,
    handlers=[logging.FileHandler(log_filepath, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class DummyTimer:
    def start(self):
        return

    def stop(self):
        return


class TvAutomation:
    def __init__(self, net_path=None):
        self.on_jevois = "pyjevois" in globals()

        # Configured camera sensor resolution. Ensure it's the same as the one in videomappings.cfg
        # FIXME: Maybe set this with an actual image
        self.original_height, self.original_width = (480, 640)

        # Resized image height passed to network
        self.target_height, self.target_width = TARGET_SIZE

        # Value scaling factor applied to input pixels
        self.scale = 1.0 / 255

        # FIXME: When I use the mean I get no identifications.
        # self.mean = [123.68, 116.77, 103.93]
        self.mean = [0, 0, 0]
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

    def get_resize_ratio(self):
        """ """
        min_length = min([self.original_height, self.original_width])
        h_ratio = min_length / self.target_height
        w_ratio = min_length / self.target_width
        return h_ratio, w_ratio

    def preprocess_image(self, image) -> Tuple[np.ndarray, Tuple]:
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

        cropped_image, top_left_x, top_left_y = center_crop(image)
        h_ratio, w_ratio = self.get_resize_ratio()

        resized_image = cv2.resize(
            cropped_image,
            dsize=(self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

        # Parameters to convert object coordinates from the resized image back to the original image
        inv_transform_params = ((top_left_x, top_left_y), (h_ratio, w_ratio))

        return resized_image, inv_transform_params

    def detect_objects(self, image):
        """
        Perform detect_objection on the input image using a pre-trained detection model and extract bounding boxes from
        the model's outputs.

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
        results = get_prediction_results(outputs, classes=self.classes)
        return results

    def draw_text(self, image, text, color="green"):
        """ """
        return cv2.putText(
            image,
            text=text,
            org=(
                3,
                self.original_height - 6,
            ),  # Bottom-left corner of the text string in the image.
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=self.colors.get(color, "white"),
        )

    def store_image(self, image, subfolder):
        """ """
        folder = Path("/", "jevois", "data", subfolder)
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"image_{counter():08d}.png"
        filepath = str(folder / filename)
        cv2.imwrite(filepath, img=image)

    def classical_cv_solution(
        self,
        im_orig,
        im_resized,
        out_frame,
        object_detection_results,
        inv_transform_params,
    ):
        """ """

        # Extract detection results
        box, score, selected_app = object_detection_results[0]

        # Convert bounding box to original image in top-left, bottom-right format
        bbox_tl, bbox_br = calc_bbox_tlbr_coords(box, inv_transform_params)

        # Process image
        im_patch = crop_image_patch(im_orig, bbox_tl, bbox_br)
        im_patch_illum = equalize_light(im_patch, "lab", eq_method="CLAHE")
        im_patch_gray = cv2.cvtColor(im_patch_illum, cv2.COLOR_BGR2GRAY)
        im_patch_edges = cv2.Canny(im_patch_gray, threshold1=50, threshold2=220)

        im_orig_drawn = im_orig.copy()

        try:
            # Attempt to call get_hough_lines up to N times, adjusting the `n_std` parameter on each attempt
            # The `n_std` parameter determines the range around the mean angle (`mean_theta`) to include lines.
            # Initially, the range is narrow, but with each retry, `n_std` is increased by some % to broaden the range.
            # This approach helps to ensure that more lines are considered if the initial attempts fail to find
            # sufficient lines.
            filtered_lines, mean_lines_angle = call_while_error(
                get_hough_lines,
                max_retries=3,
                n_std=1.0,
                varying_parameter="n_std",
                factor=1.25,
                edges=im_patch_edges,
            )

            # The factor to vary is the template scale for template matching. It will vary up to N times until the
            # desired number of peaks is found.
            selected_app_v2 = call_while_error(
                identify_selected_app_v2,
                max_retries=5,
                varying_parameter="scale",
                factor=1.05,
                im_patch=im_patch,
                apps=self.apps,
                mean_lines_angle=mean_lines_angle,
                scale=0.85,
            )

            im_orig_drawn = self.draw_text(im_orig_drawn, selected_app_v2, "green")
            self.store_image(im_orig, subfolder=f"detected/{selected_app_v2}")

        except Exception as e:
            logger.exception("Couldn't identify selected app.")
            im_orig_drawn = self.draw_text(im_orig_drawn, "Error detecting app", "red")
            self._process_and_display(im_orig, im_orig_drawn, out_frame, "errors")

            if not self.on_jevois and DEBUG:
                im_orig_drawn = draw_boxes(im_orig, [box], inv_transform_params)
                im_resized_drawn = draw_boxes(im_resized, [box])
                im_patch_drawn = draw_hough_lines(im_patch, filtered_lines)
                cv2.imshow(f"Boxes on scaled:", im_resized_drawn)
                cv2.imshow(f"im_patch", im_patch)
                cv2.imshow("im_patch_illum", im_patch_illum)
                cv2.imshow("im_patch_gray", im_patch_gray)
                cv2.imshow("im_patch_edges", im_patch_edges)
                cv2.waitKey(0)

        return

    def _process_and_display(self, image_original, image_drawn, out_frame, subfolder):
        """Process and display the image based on the platform."""

        if self.on_jevois:
            self.store_image(image_original, subfolder=subfolder)
            out_frame.sendCv(image_drawn)
        elif DEBUG:
            cv2.imshow("Detections", image_drawn)
            cv2.waitKey(0)

    def handle_no_detections(self, im_original, out_frame):
        """ """

        fps = self.timer.stop()
        logger.debug(f"No detections. FPS={fps}.")

        im_drawn = im_original.copy()
        im_drawn = self.draw_text(im_drawn, "No apps detected", color="white")
        self._process_and_display(im_original, im_drawn, out_frame, "no_detected")

    def handle_detections(
        self, im_orig, object_detection_results, inv_transform_params, out_frame
    ):
        im_drawn = im_orig.copy()

        # Extract detection results
        box, score, selected_app = object_detection_results[0]

        # Draw the detection results on the original image
        im_drawn = self.draw_text(im_drawn, selected_app)
        im_drawn = draw_boxes(im_drawn, [box], inv_transform_params)

        fps = self.timer.stop()
        logger.debug(f"Finished detection. FPS={fps}.")

        self._process_and_display(
            im_orig, im_drawn, out_frame, f"detected/{selected_app}"
        )

    def process(self, in_frame, out_frame):
        """
        Process the input frame for object detection and draw results on the output frame.

        Parameters:
        - in_frame: The input frame containing the image data.
        - out_frame: The output frame to send the processed image.
        """

        logger.debug("Starting detection")
        self.timer.start()

        # Obtain the image depending on the source
        im_orig = in_frame.getCvBGR() if self.on_jevois else in_frame

        # Skip processing if the image is too blurry
        if calc_blur(im_orig) < MIN_VARIANCE_OF_LAPLACIAN:
            logger.debug("Blurry image")
            self.timer.stop()
            return

        # Preprocess the image and get transformation parameters
        im_resized, inv_transform_params = self.preprocess_image(im_orig)

        # Detect objects in the preprocessed image
        object_detection_results = self.detect_objects(im_resized)

        if not object_detection_results:
            self.handle_no_detections(im_orig, out_frame)

        else:
            self.handle_detections(
                im_orig, object_detection_results, inv_transform_params, out_frame
            )
            # Add the Classical CV solution
            self.classical_cv_solution(
                im_orig,
                im_resized,
                out_frame,
                object_detection_results,
                inv_transform_params,
            )


if __name__ == "__main__":
    net_path = "runs/detect/train18/weights/best.onnx"
    automation = TvAutomation(net_path)

    TARGET_DIR = Path("data", "originals")
    for filename in TARGET_DIR.rglob("*.png"):
        # cv2.destroyAllWindows()
        print(f"Analyzing file {filename}")
        im_orig = cv2.imread(str(filename))
        automation.process(im_orig, None)

        # if self.on_jevois:
        #     jevois.sendSerial(f"Detected {len(boxes)} boxes")

        # Gotta add the IR sensor for using this
        # self.filter.measurement_update(selected_app)
        # print(f"Filtered app = {self.filter.get_prediction()}")
