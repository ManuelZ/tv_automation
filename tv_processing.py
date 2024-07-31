# Standard Library imports
from math import degrees, cos, sin
from pathlib import Path

# External imports
import cv2
import numpy as np
from scipy.stats import mode


def center_crop(image: np.ndarray):
    """
    Crop a square region from the center of the input image.

    Extract a square crop from the center of the input image. The size of the crop is determined by the smaller
    dimension of the input image (height or width). The resulting crop will have the same width and height.

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


def get_boxes(model_outputs: np.ndarray):
    """Extract bounding boxes from model outputs using non-maximum suppression."""

    outputs = cv2.transpose(model_outputs[0])  # n_rows x n_classes
    rows, cols = outputs.shape

    boxes = []
    scores = []
    for i in range(rows):
        classes_scores = outputs[i][4:]
        # Only one class
        (min_score, max_score, _, _) = cv2.minMaxLoc(classes_scores)

        if max_score >= 0.25:
            box = box_center_format_to_corner_format(outputs[i])
            boxes.append(box)
            scores.append(max_score)

    nms_boxes_indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=0.25, nms_threshold=0.45, eta=0.5
    )

    return [boxes[i] for i in nms_boxes_indices]


def draw_boxes(image, boxes, h_scale=1.0, w_scale=1.0, x_trans=0.0, y_trans=0.0):
    """
    Boxes have to be represented the using the coordinates of the top-left corner
    along with the width and height.
    """

    im = image.copy()
    for box in boxes:
        top_left_x, top_left_y, width, height = box
        cv2.rectangle(
            im,
            pt1=(
                int(x_trans + top_left_x * h_scale),
                int(y_trans + top_left_y * w_scale),
            ),  # fmt:skip
            pt2=(
                int(x_trans + (top_left_x + width) * h_scale),
                int(y_trans + (top_left_y + height) * w_scale),
            ),
            color=(0, 255, 0),
            thickness=1,
        )
    return im


def get_image_patch(image, top_left, bottom_right):
    """Extract a rectangular patch from an image."""

    x1, y1 = top_left
    x2, y2 = bottom_right

    patch = image[y1:y2, x1:x2]

    return patch


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


def draw_hough_line(image, r, theta, color=(0, 0, 255)):
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


def draw_lines_on_image(image, lines, color=(0, 0, 255), thickness=1):
    """
    Draw lines on the input image. The lines are represented by two points in Cartesian coordinates.
    """
    im = image.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.line(
            im,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return im


def equalize(image, eq_method="CLAHE"):
    """ """
    if eq_method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        return cv2.equalizeHist(image)


def equalize_luminance(image, color_space="hsv", eq_method=None):
    """ """

    if color_space == "YCrCb":
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(image_ycrcb)
        y = equalize(y, eq_method)
        equalized_image = cv2.merge([y, cr, cb])
        improved_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCrCb2BGR)

    elif color_space == "hsv":
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        v = equalize(v, eq_method)
        equalized_image = cv2.merge([h, s, v])
        improved_image = cv2.cvtColor(equalized_image, cv2.COLOR_HSV2BGR)

    elif color_space == "lab":
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(image_lab)
        l = equalize(l, eq_method)
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


def rotate_image(image, R):
    """Rotate the image using the given rotation matrix."""

    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    return cv2.warpAffine(image, R, (w, h))


def get_hough_lines(edges, n_std=0.25, threshold=80):
    """ """

    lines = cv2.HoughLines(
        edges, rho=1, theta=np.pi / 360, srn=0, stn=0, threshold=threshold
    )
    if lines is None:
        raise Exception("Not lines found")

    # Extract theta values from the detected lines
    thetas = [line[0][1] for line in lines]

    # Calculate mean and standard deviation of thetas
    mean_theta = np.mean(thetas)
    std_theta = np.std(thetas)

    # Filter lines that are within n standard deviations of the mean of theta
    filtered_lines = lines
    if n_std is not None:
        filtered_lines = [
            line for line in lines if abs(line[0][1] - mean_theta) <= n_std * std_theta
        ]

    if len(filtered_lines) < 2:
        print(filtered_lines)
        raise Exception("Not enough lines found")

    return filtered_lines, mean_theta


def rotate_lines_to_horizontal(lines, im_h, im_w):
    """
    Rotate the given Hough lines and return their two-point representations.
    """

    point_pairs = []

    for line in lines:
        rho, theta = line[0]
        (x1, y1), (x2, y2) = hough_line_to_points(rho, theta)

        cx = im_w / 2
        cy = im_h / 2

        angle = -(90 - degrees(theta))

        R = cv2.getRotationMatrix2D((cx, cy), angle=angle, scale=1)

        pts = np.array([[x1, y1], [x2, y2]])
        pts = pts.reshape((-1, 1, 2))
        pts = cv2.transform(pts, R)

        # The rotated points
        x1, y1, x2, y2 = pts.reshape(-1)

        point_pairs.append([(x1, y1), (x2, y2)])

    return point_pairs


def identify_selected_app(image, max_y, margin_x, num_segments=9):
    """ """

    masked_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract the bottom part of the image with some margin around the sides
    lines = masked_image_gray[max_y:, margin_x:-margin_x]

    # Create a BGR image
    stacked = np.stack([lines, lines, lines], dtype=np.uint8)
    im_lines = np.moveaxis(stacked, 0, 2)
    im_lines = im_lines.copy()  # https://stackoverflow.com/a/74221232/1253729
    h, w, c = im_lines.shape

    # N lists with mean pixel values per row
    segments = np.array_split(lines, num_segments, axis=1)

    if DEBUG:
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

    mean_line_values = [s.mean(axis=1) for s in segments]
    results = np.argmax(np.vstack(mean_line_values).T, axis=1)
    index = mode(results[:5]).mode

    return APPS[index]


def preprocess_image(image, target_size):
    """
    Preprocesses an image by cropping a centered square region and resizing it to a target size.

    This function performs the following steps:
    1. Crops a square region from the center of the input image.
    2. Resizes the cropped image to the specified target size while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): The input image array with shape (height, width, channels).
        target_size (tuple): The desired output size for the image in the format (height, width).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The cropped and resized image with shape (target_size[0], target_size[1], channels).
            - float: The x-coordinate of the top-left corner of the cropped region in the original image.
            - float: The y-coordinate of the top-left corner of the cropped region in the original image.
            - float: The ratio of the target height to the smaller dimension of the original image.
            - float: The ratio of the target width to the smaller dimension of the original image.
    """

    h, w, _ = image.shape
    h_ratio, w_ratio = np.array(target_size) / min([h, w])

    cropped_image, top_left_x, top_left_y = center_crop(image)

    resized_image = cv2.resize(
        cropped_image,
        dsize=(target_size[1], target_size[0]),  # Ensure correct dimensions
        # fx=w_ratio,
        # fy=h_ratio,
        interpolation=cv2.INTER_AREA,
    )
    return resized_image, top_left_x, top_left_y, h_ratio, w_ratio


def infer(net, image):
    """
    Perform inference on the input image using a pre-trained model and extract bounding boxes from the model's outputs.

    Args:
        net (cv2.dnn_Net): The pre-trained deep learning model in OpenCV's DNN module.
        image (numpy.ndarray): The input image array with shape (height, width, channels).

    Returns:
        list: A list of bounding boxes predicted by the model. Each box is typically represented as a list or array
              containing coordinates and dimensions, depending on the implementation of the `get_boxes` function.
    """
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255,
        size=TARGET_SIZE,
        swapRB=True,
        mean=(0, 0, 0),  # Adjust mean values if necessary
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward()
    return get_boxes(outputs)


def get_min_max_y(lines):
    """Calculate the minimum and maximum y-coordinates from a list of line segments."""

    if len(lines) < 2:
        raise Exception("Not enough lines abailable.")

    ys = [p1[1] for p1, p2 in lines]
    min_y, max_y = min(ys), max(ys)

    return min_y, max_y


def compute_bounding_box_coordinates(box, top_left_x, top_left_y, h_ratio, w_ratio):
    """
    Compute the top-left and bottom-right coordinates of a bounding box in the original image.

    Args:
        box (tuple): A tuple representing the bounding box, with (x, y, width, height).
        top_left_x (int): The x-coordinate of the top-left corner of the bounding box in the scaled image.
        top_left_y (int): The y-coordinate of the top-left corner of the bounding box in the scaled image.
        h_ratio (float): The height scaling ratio between the scaled and original image.
        w_ratio (float): The width scaling ratio between the scaled and original image.

    Returns:
        tuple: A tuple containing two tuples:
            - The coordinates of the top-left corner (x, y) of the bounding box in the original image.
            - The coordinates of the bottom-right corner (x, y) of the bounding box in the original image.

    """
    # Extract box coordinates and dimensions
    x, y, width, height = box

    # Calculate top-left and bottom-right coordinates in the original image
    top_left = (
        int(top_left_x + x * (1 / w_ratio)),
        int(top_left_y + y * (1 / h_ratio)),
    )
    bottom_right = (
        int(top_left_x + (x + width) * (1 / w_ratio)),
        int(top_left_y + (y + height) * (1 / h_ratio)),
    )

    return top_left, bottom_right


if __name__ == "__main__":
    TARGET_SIZE = (256, 256)
    MARGIN_X = 10
    MODEL_PATH = Path("runs", "detect", "train8", "weights", "best.onnx")
    NET = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    DEBUG = True

    APPS = {
        0: "Youtube",
        1: "Television",
        2: "Netflix",
        3: "Max",
        4: "Internet",
        5: "Prime video",
        6: "TV en vivo",
        7: "Movistar TV App",
        8: "Spotify",
    }

    for filename in Path("data", "originals").rglob("*.png"):
        image_original = cv2.imread(str(filename))

        # Crop a centered square region and resizing it to a target size
        image_resized, trans_x, trans_y, h_ratio, w_ratio = preprocess_image(
            image_original, TARGET_SIZE
        )
        boxes = infer(NET, image_resized)
        top_left, bottom_right = compute_bounding_box_coordinates(
            boxes[0], trans_x, trans_y, h_ratio, w_ratio
        )

        image_resized_drawn = draw_boxes(image_resized, boxes)
        image_original_drawn = draw_boxes(
            image_original, boxes, (1 / h_ratio), (1 / w_ratio), trans_x, trans_y
        )

        # Process image
        im_patch = get_image_patch(image_original, top_left, bottom_right)
        im_patch_illuminated = equalize_luminance(im_patch, "lab", eq_method="CLAHE")
        im_patch_gray = cv2.cvtColor(im_patch_illuminated, cv2.COLOR_BGR2GRAY)
        im_patch_blurred = cv2.GaussianBlur(im_patch_gray, (3, 3), sigmaX=0, sigmaY=0)
        im_patch_edges = cv2.Canny(im_patch_blurred, threshold1=50, threshold2=220)

        try:
            # Identify lines
            filtered_lines, mean_theta = get_hough_lines(im_patch_edges, n_std=1)
        except Exception as e:
            cv2.imshow("Error", im_patch_edges)
            print(e)
            cv2.waitKey(0)
            cv2.destroyWindow("Error")
            continue

        im_patch_drawn = draw_hough_lines(im_patch, filtered_lines)

        # Align images to horizontal
        rot_angle = -(90 - degrees(mean_theta))
        im_h, im_w, _ = im_patch.shape
        rot_mat = create_rot_mat(rot_angle, im_h, im_w)
        im_patch_rotated = rotate_image(im_patch, rot_mat)
        rotated_lines = rotate_lines_to_horizontal(filtered_lines, im_h, im_w)
        im_patch_rotated_drawn = draw_lines_on_image(im_patch_rotated, rotated_lines)

        # Create a mask
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        min_y, max_y = get_min_max_y(rotated_lines)
        mask[min_y:max_y, MARGIN_X:-MARGIN_X] = 255
        kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=6)

        im_patch_masked = cv2.bitwise_and(im_patch_rotated, im_patch_rotated, mask=mask)

        selected_app = identify_selected_app(im_patch_masked, max_y, margin_x=MARGIN_X)

        print(f"Selected app is: {selected_app}")

        if DEBUG:
            cv2.imshow(f"Boxes on scaled:", image_resized_drawn)
            cv2.imshow(f"Boxes on original", image_original_drawn)
            cv2.imshow(f"Image patch", im_patch)
            cv2.imshow("Patch - Equalized LAB", im_patch_illuminated)
            cv2.imshow("Patch - Gray after equalization", im_patch_gray)
            cv2.imshow("Patch - Gray after blurring", im_patch_blurred)
            cv2.imshow("Patch - Edges", im_patch_edges)
            cv2.imshow("Patch - Lines", im_patch_drawn)
            cv2.imshow("Patch - Rotated", im_patch_rotated)
            cv2.imshow("Patch - Rotated with lines", im_patch_rotated_drawn)
            cv2.imshow("Mask", mask)
            cv2.imshow("Patch - Masked", im_patch_masked)
            cv2.waitKey(0)
