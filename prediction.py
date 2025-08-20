import cv2
from ultralytics import YOLO
import os
from collections import defaultdict, Counter, deque
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
import torch
from PIL import Image
import streamlit as st
from vit_pytorch import ViT

model = YOLO("D:/yolo8_congTruong/model/best.pt")  # Load YOLO model
# ===== Load models =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

@st.cache_resource
def load_models():
    yolo_model = YOLO("D:/yolo8_congTruong/model/best.pt")
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=2,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.load_state_dict(torch.load("D:/yolo8_congTruong/train_model/vit_helmet_best.pth", map_location="cpu"))
    model.eval()
    return yolo_model, model

def is_helmet_on_head(person_box, helmet_boxes):
    px1, py1, px2, py2 = person_box
    head_top = py1
    head_bottom = py1 + (py2 - py1) * 0.2
    head_left = px1
    head_right = px2
    for hx1, hy1, hx2, hy2 in helmet_boxes:
        if max(hx1, head_left) < min(hx2, head_right) and max(hy1, head_top) < min(hy2, head_bottom):
            return True
    return False

def predict_video(input_path, output_path):
    yolo_model, vit_model = load_models()
    conf_threshold = 0.2
    alerts = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Không mở được video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_no_helmet = 0
    has_person = False
    frame_number = 0
    last_alert_time = -1  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_number += 1

        timestamp = frame_number / fps

        results = yolo_model(frame, verbose=False)[0]
        persons, helmets = [], []

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            label = results.names[cls].lower()
            xyxy = box.xyxy.cpu().numpy()[0]
            if label == "person":
                persons.append(xyxy)
            elif label in ("helmet"):
                helmets.append(xyxy)

        no_helmet_count = 0
        for p_box in persons:
            has_person = True
            has_helmet = is_helmet_on_head(p_box, helmets)
            x1, y1, x2, y2 = map(int, p_box)

            # Giới hạn toạ độ trong khung hình
            x1_c = max(0, x1)
            y1_c = max(0, y1)
            x2_c = min(frame.shape[1], x2)
            y2_c = min(frame.shape[0], y2)

            color = (0, 255, 0)
            label_text = "Helmet"

            if not has_helmet and (x2_c > x1_c and y2_c > y1_c):
                crop_img = frame[y1_c:y2_c, x1_c:x2_c]
                if crop_img is not None and crop_img.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        pred = torch.argmax(vit_model(img_tensor), dim=1).item()
                    if pred == 0:  # without helmet
                        no_helmet_count += 1
                        label_text = "No Helmet"
                        color = (0, 0, 255)

            # Vẽ khung và label
            cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), color, 2)
            cv2.putText(frame, label_text, (x1_c, y1_c - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        num_no_helmet += no_helmet_count

        # Lưu alert nếu phát hiện người không đội mũ
        if no_helmet_count > 0:
            current_second = int(timestamp)
            if current_second != last_alert_time:
                alerts.append({
                    "frame_number": frame_number,
                    "timestamp": round(timestamp, 2),
                    "num_no_helmet": no_helmet_count
                })
                last_alert_time = current_second

        out.write(frame)

    cap.release()
    out.release()

    if not has_person:
        return "Không có người nào trong video.", None

    return alerts





