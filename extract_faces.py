from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import os

mtcnn = MTCNN(image_size=160, margin=0)

def extract_faces(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for i, box in enumerate(boxes):
                face = frame.crop(box)
                face.save(f'{save_dir}/frame{frame_count}_face{i}.jpg')

    cap.release()
