"""Main module."""

import cv2
import numpy as np
import pandas as pd
import layoutparser as lp
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from paddleocr import PaddleOCR, draw_ocr

class TableOCR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.ocr = PaddleOCR(lang='en')
        self.model = lp.AutoLayoutModel("lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e", label_map={0: "Table"})

    def extract_table(self):
        image = cv2.imread(self.image_path)
        layout = self.model.detect(image)
        for l in layout:
            if l.type == 3:
                x_1 = int(l.block.x_1)
                y_1 = int(l.block.y_1)
                x_2 = int(l.block.x_2)
                y_2 = int(l.block.y_2)
                break
        table_image = image[y_1:y_2, x_1:x_2]
        cv2.imwrite('./tableocr/im1.jpg', table_image)

    def detect_and_recognize_text(self):
        output = self.ocr.ocr('./tableocr/im1.jpg')
        return output

    def preprocess_image(self, image_path):
        img_vert = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_cv = cv2.imread(image_path)
        thresh, img_bin = cv2.threshold(img_vert, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        kernel_len = np.array(img_vert).shape[1] // 100
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical = cv2.dilate(image_1, ver_kernel, iterations=3)
        cnts = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        steps = []
        for c in cnts:
            if abs(c[0][0][1] - c[1][0][1]) > (image_cv.shape[0] * 0.5):
                steps.append(c[0][0][0])
        steps.sort()
        vert_lines = 0
        for c in steps:
            image_cv = np.insert(image_cv, c + vert_lines * 30, np.zeros([30, image_cv.shape[0], 3], dtype=np.uint8) + 255, axis=1)
            vert_lines += 1
        cv2.imwrite('./tableocr/im3.jpg', image_cv)
        return image_cv

    def draw_boxes(self, image_cv, output):
        boxes = [line[0] for line in output]
        texts = [line[1][0] for line in output]
        image_boxes = image_cv.copy()
        return image_boxes

    def get_horizontal_and_vertical_lines(self, boxes, image_cv):
        im = image_cv.copy()
        horiz_boxes = []
        vert_boxes = []
        for box in boxes:
            x_h, x_v = 0, int(box[0][0])
            y_h, y_v = int(box[0][1]), 0
            width_h, width_v = image_cv.shape[1], int(box[2][0] - box[0][0])
            height_h, height_v = int(box[2][1] - box[0][1]), image_cv.shape[0]
            horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
            vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])
        return horiz_boxes, vert_boxes

    def non_max_suppression(self, horiz_boxes, vert_boxes, probabilities, image_cv):
        horiz_out = cv2.dnn.NMSBoxes(
                                        horiz_boxes,
                                        probabilities,
                                        score_threshold=0.0,  
                                        nms_threshold=0.85,     
                                        eta=1.0,              
                                        top_k=1000             
                                    )
        horiz_lines = np.sort(np.array(horiz_out))
        im_nms = image_cv.copy()
        for val in horiz_lines:
            cv2.rectangle(im_nms, (int(horiz_boxes[val][0]), int(horiz_boxes[val][1])),
                          (int(horiz_boxes[val][2]), int(horiz_boxes[val][3])), (0, 0, 255), 1)
        vert_out = cv2.dnn.NMSBoxes(
                                        vert_boxes,
                                        probabilities,
                                        score_threshold=0.0,   
                                        nms_threshold =0.85,     
                                        eta=1.0,              
                                        top_k=1000            
                                    )
        vert_lines = np.sort(np.array(vert_out))
        tmp = vert_lines.copy()
        for i in vert_lines:
            if (abs(vert_boxes[i][2] - vert_boxes[i][0]) < 10) or (abs(vert_boxes[i][1] - vert_boxes[i][3]) < 10):
                tmp.remove(i)
        vert_lines = tmp
        return horiz_lines, vert_lines

    def convert_to_csv(self, horiz_lines, vert_lines, boxes, texts, vert_boxes, horiz_boxes):
        out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
        unordered_boxes = []
        for i in vert_lines:
            unordered_boxes.append(vert_boxes[i][0])
        ordered_boxes = np.argsort(unordered_boxes)

        def intersection(box_1, box_2):
            return [box_2[0], box_1[1], box_2[2], box_1[3]]

        def iou(box_1, box_2):
            x_1 = max(box_1[0], box_2[0])
            y_1 = max(box_1[1], box_2[1])
            x_2 = min(box_1[2], box_2[2])
            y_2 = min(box_1[3], box_2[3])
            inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
            if inter == 0:
                return 0
            box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
            box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
            return inter / float(box_1_area + box_2_area - inter)

        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                    if (iou(resultant, the_box) > 0.1) and (texts[b] != out_array[i][j - 1]):
                        out_array[i][j] = texts[b]
        out_array = np.array(out_array).squeeze()
        print(out_array)
        out =  pd.DataFrame(out_array)
        out.dropna(how='all', axis=1, inplace=True) 
        out.dropna(how='all', axis=0, inplace=True) 
        out.to_csv('./tableocr/sample.csv')

    def extract(self):
        self.extract_table()
        output = self.detect_and_recognize_text()
        image_cv = self.preprocess_image(self.image_path)
        boxes = [item[0] for item in output[0]]
        texts = [item[1][0] for item in output[0]]
        horiz_boxes, vert_boxes = self.get_horizontal_and_vertical_lines(boxes, image_cv)
        probabilities = [line[1][1] for line in output[0]]
        horiz_lines, vert_lines = self.non_max_suppression(horiz_boxes, vert_boxes, probabilities, image_cv)
        self.convert_to_csv(horiz_lines, vert_lines, boxes, texts, vert_boxes, horiz_boxes)

if __name__ == "__main__":
    ocr = TableOCR("./tableocr/TABLE8.png")
    ocr.extract()
