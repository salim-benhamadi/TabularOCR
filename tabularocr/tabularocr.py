import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import layoutparser as lp
import os

class tabularocr:
    def __init__(self, image_path):
        self.image_path = image_path
        self.ocr = PaddleOCR(lang='en')
        self.model = lp.AutoLayoutModel("lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e", label_map={0: "Table"})

    def extract_tables(self):
        if self.image_path.lower().endswith('.pdf'):
            pages = convert_from_path(self.image_path)
            table_images = []
            for page in pages:
                image_np = np.array(page)
                tables = self.model.detect(image_np)
                for table in tables:
                    if table.type == 3:
                        table_image = image_np[int(table.block.y_1):int(table.block.y_2), int(table.block.x_1):int(table.block.x_2)]
                        table_images.append(table_image)
            return table_images
        else:
            image = cv2.imread(self.image_path)
            layout = self.model.detect(image)
            table_images = []
            for table in layout:
                if table.type == 3:
                    table_image = image[int(table.block.y_1):int(table.block.y_2), int(table.block.x_1):int(table.block.x_2)]
                    table_images.append(table_image)
            return table_images

    def detect_and_recognize_text(self, table_image):
        output = self.ocr.ocr(table_image)
        return output[0]

    def preprocess_image(self, image):
        img_vert = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, img_bin = cv2.threshold(img_vert, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        kernel_len = np.array(img_vert).shape[1] // 100
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical = cv2.dilate(image_1, ver_kernel, iterations=3)
        cnts, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        steps = [c[0][0][0] for c in cnts if abs(c[0][0][1] - c[1][0][1]) > (image.shape[0] * 0.5)]
        steps.sort()
        for c in steps:
            image = np.insert(image, c, np.zeros([30, image.shape[0], 3], dtype=np.uint8) + 255, axis=1)
        return image

    def convert_to_format(self, table_data, output_path, export_format):
        for table_id, (boxes, texts, vert_lines, horiz_lines, vert_boxes, horiz_boxes) in enumerate(table_data):
            out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]
            ordered_boxes = np.argsort([vert_boxes[i][0] for i in vert_lines])

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

            for i, horiz_line in enumerate(horiz_lines):
                for j, vert_line in enumerate(vert_lines):
                    resultant = intersection(horiz_boxes[horiz_line], vert_boxes[vert_lines[ordered_boxes[j]]])
                    for b, box in enumerate(boxes):
                        the_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
                        if (iou(resultant, the_box) > 0.1) and (texts[b] != out_array[i][j - 1]):
                            out_array[i][j] = texts[b]

            out_array = np.array(out_array).squeeze()
            non_empty_columns = ~(out_array == '').all(axis=0)
            out_array = out_array[:, non_empty_columns]
            out = pd.DataFrame(out_array)
            out = out.rename(columns=out.iloc[0]).drop(out.index[0])
            out = out.T.drop_duplicates().T

            if export_format.lower() == 'csv':
                out.to_csv(os.path.join(output_path, f'table_{table_id}.csv'), index=False)
            elif export_format.lower() == 'excel':
                out.to_excel(os.path.join(output_path, f'table_{table_id}.xlsx'), index=False)
            else:
                print(f"Unsupported export format '{export_format}'. Skipping table {table_id} export.")

    def get_horizontal_and_vertical_lines(self, boxes, image):
        horiz_boxes = [[0, int(box[0][1]), image.shape[1], int(box[2][1])] for box in boxes]
        vert_boxes = [[int(box[0][0]), 0, int(box[2][0]), image.shape[0]] for box in boxes]
        return horiz_boxes, vert_boxes

    def non_max_suppression(self, horiz_boxes, vert_boxes, probabilities, image):
        horiz_lines = [i[0] for i in enumerate(probabilities)]
        vert_lines = [i[0] for i in enumerate(probabilities)]

        horiz_out = cv2.dnn.NMSBoxes(
            horiz_boxes,
            probabilities,
            score_threshold=0.0,
            nms_threshold=0.80,
            eta=1.0,
            top_k=1000
        )

        vert_out = cv2.dnn.NMSBoxes(
            vert_boxes,
            probabilities,
            score_threshold=0.0,
            nms_threshold=0.95,
            eta=1.0,
            top_k=1000
        )

        horiz_lines = np.sort(np.array(horiz_out))
        vert_lines = np.sort(np.array(vert_out))

        vert_lines = [i for i in vert_lines if (abs(vert_boxes[i][2] - vert_boxes[i][0]) >= 10)
                      and (abs(vert_boxes[i][1] - vert_boxes[i][3]) >= 10)]

        return horiz_lines, vert_lines

    def extract(self, output_path='output', export_format='csv'):
        os.makedirs(output_path, exist_ok=True)
        table_images = self.extract_tables()
        table_data = []
        for table_image in table_images:
            output = self.detect_and_recognize_text(table_image)
            boxes = [item[0] for item in output]
            texts = [item[1][0] for item in output]
            horiz_boxes, vert_boxes = self.get_horizontal_and_vertical_lines(boxes, table_image)
            probabilities = [line[1][1] for line in output]
            horiz_lines, vert_lines = self.non_max_suppression(horiz_boxes, vert_boxes, probabilities, table_image)
            table_data.append((boxes, texts, vert_lines, horiz_lines, vert_boxes, horiz_boxes))

        if export_format.lower() == 'csv':
            self.convert_to_format(table_data, output_path, 'csv')
        elif export_format.lower() == 'excel':
            self.convert_to_format(table_data, output_path, 'excel')
        else:
            print("Unsupported export format. Please choose 'csv' or 'excel'.")

if __name__ == "__main__":
    ocr = tabularocr("tabularocr\TABLE8.png")
    ocr.extract(export_format='excel')
