# utils/data_preprocessing.py

import os
from pathlib import Path

import cv2
import numpy as np


def load_data(dataset_dir, emotion_labels, image_size=(48, 48)):
    images = []
    labels = []

    for dataset_type in ['train', 'test']:
        dataset_path = dataset_dir / dataset_type
        if not dataset_path.exists():
            print(f"Dataset directory not found: {dataset_path}")
            continue

        for emotion, happiness_score in emotion_labels.items():
            emotion_dir = dataset_path / emotion
            if not emotion_dir.exists():
                print(f"Emotion directory not found: {emotion_dir}")
                continue

            for img_file in emotion_dir.glob("*.*"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # 画像のリサイズ
                img = cv2.resize(img, image_size)

                # 正規化
                img = img.astype('float32') / 255.0

                images.append(img)
                labels.append(happiness_score)

    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(-1, image_size[0], image_size[1], 1)

    return images, labels
