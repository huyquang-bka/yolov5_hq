import cv2
import string
import imutils
from imutils.perspective import four_point_transform
import numpy as np
import easyocr
from detect_noargs import detect
import time
from skimage.segmentation import clear_border
import os


def add_extend(image, type="black", size=5):
    H, W = image.shape[:2]
    if type == "black":
        black_image = np.zeros((H + size * 2, W + size * 2), np.uint8)
    elif type == "white":
        black_image = np.full((H + size * 2, W + size * 2), 255, np.uint8)
    black_image[size:H + size, size:W + size] = image
    return black_image


class PlateFinder():
    def __init__(self):
        self.squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.reader = easyocr.Reader(['en'])
        pass

    def is_square_lp(self, image):
        H, W = image.shape[:2]
        if W / H < 2.5:
            return True
        return False

    def find_plate_opencv(self, image):
        H, W = image.shape[:2]
        crop = image[H // 2:, W // 10: W * 8 // 10]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.squareKern)
        thresh = cv2.threshold(light, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        keypoints = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # cv2.drawContours(crop_copy, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("contours", crop_copy)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                if w < h:
                    continue
                # cv2.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow("crop", crop)
                # cv2.waitKey()
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                warp = four_point_transform(gray, box)
                warp = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                return imutils.resize(warp, width=400)
        return None

    def find_plate_yolo(self, image):
        plate = detect(image)
        if plate is not None:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # thresh_copy = thresh.copy()
            thresh = clear_border(thresh)
            # cv2.imshow("thresh", thresh)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # if len(contours) >= 2:
            #     cnt = sorted(contours, key=cv2.contourArea, reverse=True)[:2][1]
            #     rect = cv2.minAreaRect(cnt)
            #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            #     box = np.int0(box)
            #     warp = four_point_transform(gray, box)
            #     warp = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #     warp = clear_border(warp)
            #     return imutils.resize(warp, width=400)
            # thresh_copy = cv2.cvtColor(thresh_copy, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(thresh_copy, contours, -1, (0, 0, 255), 2)
            # cv2.imshow("thresh", imutils.resize(thresh_copy, width=400))
            return imutils.resize(thresh, width=400)
        return None

    def image_to_text(self, img, allow_list=string.ascii_uppercase + string.digits):
        text = self.reader.readtext(img, detail=0, allowlist=allow_list, decoder="beamsearch",
                                    text_threshold=0.5)
        if text:
            return text[0]
        return ""

    def CCA(self, image):
        H, W = image.shape[:2]
        output = cv2.connectedComponentsWithStats(
            image, 8, cv2.CV_32S)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        (numLabels, labels, stats, centroids) = output
        bboxes = []
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w > W // 4:
                continue
            if w * h < 300:
                continue
            if H / h > 10:
                continue
            if (y <= 10 or y + h > H - 10) and (x < 10 or x + w > W - 10):
                continue
            bboxes.append([x, y, w, h])
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.imshow("image_from_CCA", image)
        return bboxes

    def bboxes_square_LP(self, image):
        bboxes = self.CCA(image)
        if not bboxes:
            return []
        y_mean = sum([i[1] for i in bboxes]) / len(bboxes)
        line1 = []
        line2 = []
        for bbox in bboxes:
            x, y, w, h = bbox
            if y < y_mean:
                line1.append([x, y, w, h])
            else:
                line2.append([x, y, w, h])
        line1 = sorted(line1)
        line2 = sorted(line2)
        bboxes_LP = line1 + line2
        return bboxes_LP

    def bboxes_rec_LP(self, image):
        bboxes = self.CCA(image)
        if not bboxes:
            return []
        return sorted(bboxes)

    def process_LP(self, image):
        if self.is_square_lp(image):
            bboxes = self.bboxes_square_LP(image)
        else:
            bboxes = self.bboxes_rec_LP(image)
        return bboxes

    def plate_recognition(self, img):
        plate = self.find_plate_yolo(img)
        if plate is None:
            return None, None
        bboxes = self.process_LP(plate)
        if not bboxes:
            return None, None
        text = ""
        if bboxes:
            for index, bbox in enumerate(bboxes):
                if index == 2:
                    allow_list = string.ascii_uppercase
                else:
                    allow_list = string.digits
                x, y, w, h = bbox
                crop = plate[y:y + h, x:x + w]
                crop = add_extend(crop, size=10)
                t = self.image_to_text(crop, allow_list=allow_list)
                text += t
        return plate, text


plate_finder = PlateFinder()
video_path = r"D:\cam_thu_vien\17_3_2022\LP\2022-03-17\LP_cut.mp4"
cap = cv2.VideoCapture(video_path)
lp_list = []
frame_count = 0
frame_no_lp = 0
check = True
while True:
    s = time.time()
    frame_count += 1
    ret, frame = cap.read()
    if lp_list:
        if frame_count % 3 != 0:
            continue
    if check:
        plate, text = plate_finder.plate_recognition(frame)
        if plate is not None:
            with open("test_lp.txt", "a+") as f:
                f.write(text + "\n")
        if frame_no_lp >= 10 and len(lp_list) > 0:
            lp_text = max(set(lp_list), key=lp_list.count)
            for fileName in os.listdir(r"D:/Lab IC/yolov5_tracking/LP"):
                os.remove("D:/Lab IC/yolov5_tracking/LP/" + fileName)
            with open(fr"D:/Lab IC/yolov5_tracking/LP/{lp_text}", "w+") as f:
                f.write(lp_text)
            frame_no_lp = 0
            lp_list = []
        if text and 6 <= len(text) <= 9:
            lp_list.append(text)
        if plate is not None:
            # cv2.imshow("plate", plate)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_no_lp = 0
            # cv2.imwrite("plate.jpg", plate)
        else:
            frame_no_lp += 1
    print("FPS: ", 1 / (time.time() - s))
    cv2.imshow("frame", imutils.resize(frame, width=960))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 32:
        cv2.waitKey()
    elif key == ord('c'):
        check = not check
    elif key == ord('n'):
        frame_count += 25 * 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    elif key == ord('p') and frame_count > 25 * 10:
        frame_count -= 25 * 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
