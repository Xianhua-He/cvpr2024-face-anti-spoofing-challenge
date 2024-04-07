import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from concurrent.futures.thread import ThreadPoolExecutor


# pip install insightface
# default model from: https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view?usp=sharing
# load buffalo_s model, 159MB


if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=32)
    base_path = "xxx"

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    delta = 20  # expand 20 pixel
    protocols = ['p1', 'p2.1', 'p2.2']

    def crop_face_func(img_full_path):
        img = cv2.imread(img_full_path)
        height, width, _ = img.shape
        if height >= 700 and width >= 700:
            try:
                faces = app.get(img)
                bbox = faces[0]['bbox']
                point1 = [int(bbox[0]), int(bbox[1])]
                point2 = [int(bbox[2]), int(bbox[3])]

                point_1 = [max(0, point1[0] - delta), max(0, point1[1] - delta)]
                point_2 = [min(point2[0] + delta, width), min(point2[1] + delta, height)]
                crop_face = img[point_1[1]:point_2[1], point_1[0]:point_2[0], :]

                # crop_face overwrites the original image
                cv2.imwrite(f"{img_full_path}", crop_face)
                print("crop face")
            except:
                print("no face")

    for protocol in protocols:
        all_data = []
        train_lines = open(f'{base_path}/cvpr2024/data/{protocol}/train_label.txt', 'r').readlines()
        dev_test_lines = open(f'{base_path}/cvpr2024/data/{protocol}/dev_test.txt', 'r').readlines()
        all_data = train_lines + dev_test_lines

        cnt = 0
        for line in all_data:
            split_arr = line.strip().split()
            cnt += 1
            print(f"cnt: {cnt}")
            img_path = split_arr[0]
            img_full_path = os.path.join(base_path, "cvpr2024/data", img_path)
            executor.submit(crop_face_func, img_full_path)
