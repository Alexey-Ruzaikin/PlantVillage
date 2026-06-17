import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Загрузка модели
model = tf.keras.models.load_model("plant_model.keras")

# Загрузка названий классов
with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]


# База рекомендаций
recommendations = {
    "Tomato_Early_blight":
        "Удалите поражённые листья и обработайте растение фунгицидом.",

    "Tomato_Late_blight":
        "Снизьте влажность и используйте медьсодержащие препараты.",

    "Tomato_healthy":
        "Растение здорово. Лечение не требуется.",

    "Potato___Early_blight":
        "Удалите заражённые листья и примените фунгицид.",

    "Potato___Late_blight":
        "Используйте препараты против фитофтороза.",

    "Potato___healthy":
        "Растение здорово.",

    "Pepper__bell___healthy":
        "Признаков заболевания не обнаружено."
}


# Функция предсказания
def predict_disease(image):
    img = image.resize((224, 224))

    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    disease = class_names[predicted_class]

    confidence = np.max(prediction) * 100

    return disease, confidence


