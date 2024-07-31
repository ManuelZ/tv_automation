import pyjevois
import libjevois as jevois
import cv2 as cv
import numpy as np


# @videomapping YUYV 320 336 15.0 YUYV 320 336 15.0 JeVois
# @email None
# @address None
# @copyright None
# @mainurl None
# @supporturl None
# @otherurl None
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class TvAutomation:
    def __init__(self):
        self.target_height = 256  # Resized image height passed to network
        self.target_width = 256  # Resized image width passed to network
        self.scale = 1.0 / 255  # Value scaling factor applied to input pixels
        # self.mean = [123.68, 116.77, 103.93]
        self.mean = [0, 0, 0]
        self.rgb = False  # True if model expects RGB inputs, otherwise it expects BGR
        self.n = 0

        # This network takes a while to load from microSD. To avoid timouts at construction,
        # we will load it in process() instead.

        self.timer = jevois.Timer("TV Automation", 10, jevois.LOG_DEBUG)

    def get_boxes(self, outputs):
        """ """

        outputs = np.array([cv.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(
                classes_scores
            )
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)

        nms_boxes_indices = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        return nms_boxes_indices, boxes

    def center_crop(self, image):
        h, w, _ = image.shape
        length = min((h, w))
        out_image = np.zeros((length, length, 3), np.uint8)

        start_x = w / 2 - length / 2
        start_y = h / 2 - length / 2

        out_image = image[
            int(start_y) : int(start_y + length),
            int(start_x) : int(start_x + length),
        ]

        return out_image

    def process(self, inframe, outframe):
        if not hasattr(self, "net"):
            self.classes = ["tv_apps"]
            self.model = "YOLOv8 ONNX"
            self.net = cv.dnn.readNet(pyjevois.share + "/dnn/custom/tv_apps.onnx", "")
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        original_image = inframe.getCvBGR()

        self.timer.start()

        image = self.center_crop(original_image)
        image = cv.resize(
            image, (self.target_width, self.target_height), interpolation=cv.INTER_AREA
        )

        blob = cv.dnn.blobFromImage(
            image,
            scalefactor=self.scale,
            size=(self.target_height, self.target_width),
            mean=self.mean,
            swapRB=self.rgb,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward()
        nms_boxes_indices, boxes = self.get_boxes(outputs)

        scale = 1

        for i in range(len(nms_boxes_indices)):
            index = nms_boxes_indices[i]
            box = boxes[index]
            cv.rectangle(
                image,
                pt1=(int(box[0] * scale), int(box[1] * scale)),
                pt2=(
                    int((box[0] + box[2]) * scale),
                    int((box[1] + box[3]) * scale),
                ),
                color=(0, 255, 0),
                thickness=1,
            )

        cv.imwrite(f"/jevois/data/image_{self.n:08d}.png", original_image)
        self.n += 1

        # Send output frame to host
        outframe.sendCv(image)
