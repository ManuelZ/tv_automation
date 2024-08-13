# Standard Library imports
from math import degrees, cos, sin
from itertools import count
import logging
from typing import Tuple

# External imports
try:
    import pyjevois
    from matplotlib import pyplot as plt
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass
import cv2
import numpy as np
import scipy.stats
import scipy.signal

# Local imports
from config import DEBUG, NUM_APPS

logger = logging.getLogger(__name__)


def counter(_count=count(1)):
    """https://stackoverflow.com/a/54715096/1253729"""
    return next(_count)


def center_crop(image: np.ndarray):
    """
    Crop a square region from the center of the input image. The size of the crop is determined by the smaller
    dimension of the input image (height or width). The resulting crop will have equal width and height.

    Args:
        image (numpy.ndarray): The input image array with shape (height, width, channels).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The cropped square image with shape (length, length, channels).
            - float: The x-coordinate of the top-left corner of the crop region in the original image.
            - float: The y-coordinate of the top-left corner of the crop region in the original image.
    """
    h, w, _ = image.shape
    length = min((h, w))

    top_left_x = (w / 2) - (length / 2)
    top_left_y = (h / 2) - (length / 2)

    out_image = image[
        int(top_left_y) : int(top_left_y + length),
        int(top_left_x) : int(top_left_x + length),
    ]

    return out_image, top_left_x, top_left_y


def box_center_format_to_corner_format(box):
    """
    Convert a bounding box representation from center format to corner format.

    The center format represents a bounding box using its center coordinates (cx, cy)
    and its width (w) and height (h). The corner format represents the bounding box
    using the coordinates of the top-left corner along with the width and height.
    """
    center_x, center_y, width, height = box[:4]
    return [center_x - (0.5 * width), center_y - (0.5 * height), width, height]


def get_prediction_results(
    model_outputs: np.ndarray,
    score_threshold=0.25,
    nms_threshold=0.45,
    eta=0.5,
    classes=None,
):
    """ """

    outputs = cv2.transpose(model_outputs[0])  # n_rows x n_classes
    rows, cols = outputs.shape

    boxes = []
    scores = []
    selected_apps = []
    for i in range(rows):
        classes_scores = outputs[i][4:]

        min_loc: Tuple[int, int]
        max_loc: Tuple[int, int]
        (min_score, max_score, min_loc, max_loc) = cv2.minMaxLoc(classes_scores)

        if max_score >= 0.25:
            box = box_center_format_to_corner_format(outputs[i])
            boxes.append(box)
            scores.append(max_score)
            selected_apps.append(classes[max_loc[1]])

    nms_boxes_indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold,
        nms_threshold,
        eta,
    )
    return [(boxes[i], scores[i], selected_apps[i]) for i in nms_boxes_indices]


def draw_boxes(image, boxes, inverse_transform_params=((0, 0), (1, 1))):
    """
    The parameter `boxes` has to be represented the using the coordinates of the top-left corner
    along with the width and height.
    """

    ((x_trans, y_trans), (h_ratio, w_ratio)) = inverse_transform_params

    im = image.copy()
    for box in boxes:
        top_left_x, top_left_y, width, height = box
        cv2.rectangle(
            im,
            pt1=(
                int(x_trans + top_left_x * h_ratio),
                int(y_trans + top_left_y * w_ratio),
            ),  # fmt:skip
            pt2=(
                int(x_trans + (top_left_x + width) * h_ratio),
                int(y_trans + (top_left_y + height) * w_ratio),
            ),
            color=(0, 255, 0),
            thickness=1,
        )
    return im


def crop_image_patch(image, top_left, bottom_right):
    """Extract a rectangular patch from an image."""

    x1, y1 = top_left
    x2, y2 = bottom_right

    return image[y1:y2, x1:x2]


def hough_line_to_points(r, theta):
    """Convert a line in Hough space to two endpoints in Cartesian coordinates."""

    c_theta = cos(theta)
    s_theta = sin(theta)

    x1 = r * c_theta - 1000 * s_theta
    x2 = r * c_theta + 1000 * s_theta

    y1 = r * s_theta + 1000 * c_theta
    y2 = r * s_theta - 1000 * c_theta

    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    return pt1, pt2


def draw_hough_line(image, r, theta, color=(0, 255, 0)):
    """Draw a Hough line in the input image."""
    pt1, pt2 = hough_line_to_points(r, theta)
    cv2.line(image, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA)


def draw_hough_lines(image, lines):
    """Draw Hough lines on a copy of the input image."""
    im = image.copy()
    for line in lines:
        rho, theta = line[0]
        draw_hough_line(im, rho, theta)
    return im


def equalize(image, eq_method="CLAHE", grid_size=8):
    """Apply histogram equalization to an image using the specified method."""

    if eq_method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
        return clahe.apply(image)
    else:
        return cv2.equalizeHist(image)


def equalize_light(image, color_space="hsv", eq_method=None, grid_size=8):
    """ """

    if color_space == "YCrCb":
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(image_ycrcb)
        y = equalize(y, eq_method, grid_size)
        equalized_image = cv2.merge([y, cr, cb])
        improved_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCrCb2BGR)

    elif color_space == "hsv":
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        v = equalize(v, eq_method, grid_size)
        equalized_image = cv2.merge([h, s, v])
        improved_image = cv2.cvtColor(equalized_image, cv2.COLOR_HSV2BGR)

    elif color_space == "lab":
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(image_lab)
        l = equalize(l, eq_method, grid_size)
        equalized_image = cv2.merge([l, a, b])
        improved_image = cv2.cvtColor(equalized_image, cv2.COLOR_Lab2BGR)

    else:
        raise Exception("color_space must be one of 'hsv','lab' or 'YCrCb'")

    return improved_image


def create_rot_mat(angle, h, w):
    """
    Create a rotation matrix to rotate an image around its center.

    Args
        angle: rotation angle in degrees
        h: image height
        w: image width
    """
    return cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle=angle, scale=1)


def rotate_image_to_horizontal(image, angle):
    """
    angle in radians
    """

    rot_angle = degrees(angle) - 90

    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    rot_mat = create_rot_mat(rot_angle, h, w)
    return cv2.warpAffine(image, rot_mat, (w, h))


def get_hough_lines(edges, n_std=1.0, threshold=80):
    """

    Returns
    theta in radians
    """

    lines = cv2.HoughLines(
        edges, rho=1, theta=np.pi / 360, srn=0, stn=0, threshold=threshold
    )
    if lines is None:
        raise Exception("No lines found")

    # Extract theta values from the detected lines
    thetas = [line[0][1] for line in lines]

    # Calculate approx theta and its std
    theta = np.median(thetas)
    std_theta = np.std(thetas)

    # Filter lines that are within n standard deviations of theta
    filtered_lines = lines
    if n_std is not None:
        filtered_lines = [
            line for line in lines if abs(line[0][1] - theta) <= n_std * std_theta
        ]

    if len(filtered_lines) < 2:
        print(filtered_lines)
        raise Exception("Not enough lines found")

    return filtered_lines, theta


def calc_bbox_tlbr_coords(box, inverse_transform_params):
    """
    Compute the top-left and bottom-right coordinates of a bounding box in the original image.

    Args:
        box (tuple): A tuple representing the bounding box, with (x, y, width, height).
        inverse_transform_params (tuple):
            top_left_x (int): The x-coordinate of the top-left corner of the bounding box in the scaled image.
            top_left_y (int): The y-coordinate of the top-left corner of the bounding box in the scaled image.
            h_ratio (float): The height scaling ratio between the scaled and original image.
            w_ratio (float): The width scaling ratio between the scaled and original image.

    Returns:
        tuple: A tuple containing two tuples:
            - The coordinates of the top-left corner (x, y) of the bounding box in the original image.
            - The coordinates of the bottom-right corner (x, y) of the bounding box in the original image.

    """

    x, y, width, height = box
    ((top_left_x, top_left_y), (h_ratio, w_ratio)) = inverse_transform_params

    # Calculate top-left and bottom-right coordinates in the original image
    top_left = (
        int(top_left_x + x * w_ratio),
        int(top_left_y + y * h_ratio),
    )
    bottom_right = (
        int(top_left_x + (x + width) * w_ratio),
        int(top_left_y + (y + height) * h_ratio),
    )

    return top_left, bottom_right


def resize(image, target_width: int):
    """"""
    h, w = image.shape[:2]
    r = target_width / w
    target_dim = (target_width, int(h * r))
    return cv2.resize(image, target_dim, interpolation=cv2.INTER_AREA)


def template_matching(template, image, scale):
    """ """
    image_output = image.copy()

    (tH, tW) = template.shape[:2]

    # Resize to image to have a target width, preserving the aspect ratio
    target_width = int(image.shape[1] * scale)
    resized = resize(image, target_width)

    # Take note of the resize ratio (original / new)
    r = image.shape[1] / float(resized.shape[1])

    # Break if the resized image is smaller than the template
    if resized.shape[0] < tH or resized.shape[1] < tW:
        raise Exception(f"Resized image is smaller than the template: r={r}")

    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

    (start_x, start_y) = (int(max_loc[0] * r), int(max_loc[1] * r))
    (end_x, end_y) = (int((max_loc[0] + tW) * r), int((max_loc[1] + tH) * r))

    logger.debug(f"Resizing image using ratio {r:.2f}")
    if DEBUG:
        cv2.rectangle(
            resized,
            (max_loc[0], max_loc[1]),
            (max_loc[0] + tW, max_loc[1] + tH),
            (0, 0, 255),
            1,
        )
        cv2.imshow("Resized image", resized)
        cv2.rectangle(image_output, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
        cv2.imshow("image_output", image_output)

    return image_output, (start_x, start_y, end_x, end_y)


def extract_masked_patch(image, angle, scale):
    """

    Args:
        angle: in radians
    """

    # TODO: Move these lines out
    im_h, im_w, _ = image.shape
    # Rotate image
    im_rotated = rotate_image_to_horizontal(image, angle)

    # Add black borders
    margin_left = margin_right = 20
    margin_top = margin_bottom = 0
    im_rotated = cv2.copyMakeBorder(
        im_rotated,
        margin_top,
        margin_bottom,
        margin_left,
        margin_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    if "pyjevois" in globals():
        template_path = pyjevois.share + "/template.png"
    else:
        template_path = "template.png"
    template = cv2.imread(template_path)

    _, (start_x, start_y, end_x, end_y) = template_matching(
        template, im_rotated, scale=scale
    )

    # Create mask
    im_h = end_y - start_y
    im_w = end_x - start_x
    mask = np.zeros((im_h, im_w), dtype=np.uint8)
    mask[start_y:end_y, :] = 255

    masked_image = im_rotated[start_y:end_y, start_x:end_x]

    return masked_image, end_y


def call_while_error(func, varying_parameter, factor, max_retries=3, **kwargs):
    """
    Repeatedly call a function with the given keyword arguments until it succeeds or the maximum number of
    retries is reached.

    After each failure, it will incrementally adjust the `n_std` parameter by multiplying it by 1.1 to modify
    the function's behavior on subsequent retries.

    """
    retries = 0
    while retries < max_retries:
        try:
            return func(**kwargs)
        except Exception as e:
            retries += 1
            kwargs[varying_parameter] *= factor
            logger.exception("")
            logger.debug(
                "    Retrying ({}/{}) with {}={:.2f}".format(
                    retries,
                    max_retries,
                    varying_parameter,
                    kwargs[varying_parameter],
                )
            )
            if retries >= max_retries:
                logger.error(f"Reached max number of retries: {max_retries}")
                raise


def variance_of_laplacian(image):
    """
    From: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # Compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def calc_blur(image):
    """
    From: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # Compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return variance_of_laplacian(image)


def identify_selected_app_v1(image, max_y, apps, num_segments=9):
    """ """

    masked_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract the bottom part of the image with some margin around the sides
    lines = masked_image_gray[max_y - 15 :, :]

    # Create a BGR image
    stacked = np.stack([lines, lines, lines])
    im_lines = np.moveaxis(stacked, 0, 2)
    im_lines = im_lines.copy()  # https://stackoverflow.com/a/74221232/1253729

    peaks, derivative = get_apps_sizes(image)
    dist_between_peaks = np.diff(peaks)
    segments = np.split(lines, peaks, axis=1)
    # N lists with mean pixel values per row
    # segments: list[np.ndarray] = np.array_split(lines, num_segments, axis=1)

    if DEBUG:
        h, w, c = im_lines.shape
        # Draw vertical green lines to visualize the segments
        for i in range(num_segments - 1):
            seg = segments[i]
            x = i * segments[0].shape[1] + seg.shape[1]
            cv2.line(
                im_lines,
                pt1=(x, 0),
                pt2=(x, h),
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        cv2.imshow("segments", im_lines)

    mean_line_values: list[np.ndarray] = [s.mean(axis=1) for s in segments]
    results: list[int] = np.argmax(np.vstack(mean_line_values).T, axis=1)

    index = int(scipy.stats.mode(results[:5]).mode)

    return apps[index]


def plot_segments(image, derivative, peaks, title):
    """ """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    if len(image.shape) == 2:
        ax[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax[0].imshow(image_rgb)

    # Plot peaks and signal
    ax[1].plot(derivative)
    ax[1].plot(peaks, derivative[peaks], "x")
    ax[1].margins(x=0)
    ax[1].grid()

    plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.show()


def get_inverted_gaussian_mask(image, sigma_ratio=8):
    """ """

    height, width = image.shape[:2]

    # Generate a Gaussian mask along the y-axis
    y = np.arange(height)
    mean = height / 2
    sigma = height / sigma_ratio  # Adjust sigma based on the ratio
    gaussian_mask = np.exp(-0.5 * ((y - mean) ** 2) / (sigma**2))

    # Invert the Gaussian mask
    inverted_mask = 1 - (gaussian_mask - np.min(gaussian_mask)) / (
        np.max(gaussian_mask) - np.min(gaussian_mask)
    )

    # Expand mask to match the image's dimensions
    inverted_mask_expanded = np.tile(inverted_mask, (width, 1)).T

    if len(image.shape) == 3:
        inverted_mask_expanded = np.dstack(
            [inverted_mask_expanded, inverted_mask_expanded, inverted_mask_expanded]
        )

    return inverted_mask_expanded


def get_apps_sizes(image):
    """ """
    image = equalize_light(image, "lab", eq_method="CLAHE")
    inverted_gauss_mask = get_inverted_gaussian_mask(image, sigma_ratio=12)

    if len(image.shape) == 3:
        derivative = np.gradient(image, axis=1)
        derivative = np.power(derivative, 2)
        derivative = derivative * inverted_gauss_mask
        derivative = np.median(derivative, axis=0)
        derivative = np.sum(derivative, axis=1)
        derivative = np.where(derivative > 0, np.log(derivative), 0)

    else:
        derivative = np.gradient(image, axis=0)
        derivative = np.power(derivative, 2)
        derivative = derivative * inverted_gauss_mask
        derivative = np.median(derivative, axis=0)

    peaks, _ = scipy.signal.find_peaks(derivative, height=0, distance=20)

    return peaks, derivative


class PeaksError(Exception):
    pass


def draw_vertical_segments(image, peaks):
    h, w = image.shape[:2]
    for i in peaks:
        cv2.line(image, (i, 0), (i, h), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


def identify_selected_app_v2(
    im_patch, apps, mean_lines_angle, scale, app_width_factor=0.8
):
    """ """

    # Extract masked patch and compute maximum y-coordinate
    im_patch_masked, _ = extract_masked_patch(im_patch, mean_lines_angle, scale)

    # Get app sizes and compute distances between peaks
    peaks, derivative = get_apps_sizes(im_patch_masked)
    dist_between_peaks = np.diff(peaks)

    # Compute the median width of detected applications and define a threshold to identify potentially small
    # applications. Applications with widths below this threshold are considered too small.
    median_app_width = np.median(dist_between_peaks)
    low_threshold = app_width_factor * median_app_width
    small_apps = dist_between_peaks < low_threshold

    draw_vertical_segments(im_patch_masked, peaks)

    num_peaks = len(peaks)
    if num_peaks < NUM_APPS + 1:
        raise PeaksError(f"Found LESS peaks than expected: {len(peaks)}")

    if num_peaks > NUM_APPS + 1:
        logger.error(
            f"Found MORE peaks than expected, discarding {len(peaks) - (NUM_APPS + 1)}"
        )
        peaks = peaks[: NUM_APPS + 1]

    if any(small_apps):
        small_widths = dist_between_peaks[small_apps]
        logger.error(
            f"Median app width={median_app_width:.2f}. "
            f"Apps widths smaller than {low_threshold:.1f} = {small_widths}"
        )

        msg = "Detected app size is too small"
        if DEBUG:
            plot_segments(im_patch_masked, derivative, peaks, msg)

        raise Exception(msg)

    app_index = np.argmax(dist_between_peaks)
    app_name = apps[app_index]

    if DEBUG:
        plot_segments(im_patch_masked, derivative, peaks, app_name)

    return app_name
