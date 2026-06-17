import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PlantDiagnosisApp:
    def __init__(self, root):
        self.root = root
        root.title("🌿 Экспертная система диагностики растений")
        root.geometry("900x750")
        root.configure(bg='#e8f5e9')

        # Загрузка модели
        self.load_model()

        self.setup_ui()

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model("plant_model.keras")
            with open("class_names.txt", "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f]
            self.model_loaded = True
            self.num_classes = len(self.class_names)
            print(f"✅ Модель загружена. Классов: {self.num_classes}")
        except Exception as e:
            self.model = None
            self.class_names = []
            self.model_loaded = False
            print(f"❌ Ошибка загрузки модели: {e}")

    def setup_ui(self):
        # Основной фрейм
        main_frame = tk.Frame(self.root, bg='#e8f5e9')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Заголовок
        title = tk.Label(main_frame, text="🌿 Экспертная система диагностики заболеваний растений",
                         font=("Arial", 18, "bold"), bg='#e8f5e9', fg='#2e7d32')
        title.pack(pady=10)

        # Подзаголовок
        subtitle = tk.Label(main_frame, text="Загрузите фотографию листа для определения заболевания",
                            font=("Arial", 11), bg='#e8f5e9', fg='#555')
        subtitle.pack(pady=5)

        # Статус модели
        if self.model_loaded:
            status_text = f"✅ Модель загружена | {self.num_classes} классов"
            status_color = '#4caf50'
        else:
            status_text = "❌ Модель не загружена. Запустите: python tray.py"
            status_color = '#f44336'

        status_label = tk.Label(main_frame, text=status_text, font=("Arial", 10),
                                bg='#e8f5e9', fg=status_color)
        status_label.pack(pady=5)

        # Рамка для кнопок
        btn_frame = tk.Frame(main_frame, bg='#e8f5e9')
        btn_frame.pack(pady=15)

        # Кнопки
        self.btn_load = tk.Button(btn_frame, text="📁 Загрузить изображение",
                                  command=self.load_image, font=("Arial", 11),
                                  bg='#2196f3', fg='white', padx=20, pady=8,
                                  cursor='hand2')
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_diagnose = tk.Button(btn_frame, text="🔍 Диагностика",
                                      command=self.diagnose, font=("Arial", 11),
                                      bg='#4caf50', fg='white', padx=20, pady=8,
                                      state=tk.DISABLED, cursor='hand2')
        self.btn_diagnose.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="🗑️ Очистить",
                                   command=self.clear_image, font=("Arial", 11),
                                   bg='#ff9800', fg='white', padx=20, pady=8,
                                   state=tk.DISABLED, cursor='hand2')
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Рамка для изображения
        image_frame = tk.LabelFrame(main_frame, text=" Загруженное изображение ",
                                    font=("Arial", 11, "bold"), bg='white',
                                    fg='#2e7d32')
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        self.image_label = tk.Label(image_frame, text="🖼️\n\nИзображение не выбрано",
                                    bg='#f5f5f5', font=("Arial", 14), fg='#999')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Рамка для результатов
        result_frame = tk.LabelFrame(main_frame, text=" Результаты диагностики ",
                                     font=("Arial", 11, "bold"), bg='#e8f5e9',
                                     fg='#2e7d32')
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        # Текстовое поле с прокруткой
        text_frame = tk.Frame(result_frame, bg='#e8f5e9')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = tk.Text(text_frame, font=("Consolas", 10), wrap=tk.WORD,
                                   height=12, bg='#ffffff', fg='#333',
                                   relief=tk.SUNKEN, borderwidth=1)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)

        self.current_path = None
        self.photo = None

        # Приветственное сообщение
        self.result_text.insert(tk.END,
                                "Добро пожаловать!\n\n1. Нажмите 'Загрузить изображение'\n2. Выберите фото листа растения\n3. Нажмите 'Диагностика'")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.current_path = path
            # Открываем и отображаем изображение
            img = Image.open(path)

            # Изменяем размер для отображения
            display_size = (350, 350)
            img_display = img.copy()
            img_display.thumbnail(display_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_display)
            self.image_label.config(image=self.photo, text="")

            # Активируем кнопки
            self.btn_diagnose.config(state=tk.NORMAL)
            self.btn_clear.config(state=tk.NORMAL)

            # Очищаем результаты
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "✅ Изображение загружено!\n\nНажмите 'Диагностика' для анализа.")

    def clear_image(self):
        self.current_path = None
        self.image_label.config(image='', text="🖼️\n\nИзображение не выбрано")
        self.btn_diagnose.config(state=tk.DISABLED)
        self.btn_clear.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Ожидание загрузки изображения...")

    def diagnose(self):
        if not self.model_loaded:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "❌ Модель не загружена.\n\nСначала запустите обучение:\npython tray.py")
            return

        if not self.current_path:
            return

        try:
            # Показать прогресс
            self.progress.pack(pady=10)
            self.progress.start()
            self.root.update()

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "🔬 Анализ изображения...\n\n")
            self.root.update()

            # Загрузка и подготовка изображения
            img = Image.open(self.current_path).resize((224, 224))
            img_array = np.array(img)

            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]

            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Предсказание
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100

            disease_name = self.class_names[predicted_class].replace('___', ' - ').replace('_', ' ')

            # Остановить прогресс
            self.progress.stop()
            self.progress.pack_forget()

            # Формирование результата
            result = f"""
╔══════════════════════════════════════════════════════════════╗
║                     РЕЗУЛЬТАТ ДИАГНОСТИКИ                    ║
╚══════════════════════════════════════════════════════════════╝

🌿 ДИАГНОЗ:
   {disease_name}

📊 УВЕРЕННОСТЬ:
   {confidence:.2f}%

{'   ⚠️  НИЗКАЯ УВЕРЕННОСТЬ - повторите диагностику' if confidence < 70 else '   ✅ ВЫСОКАЯ УВЕРЕННОСТЬ'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 РЕКОМЕНДАЦИИ:
"""

            # Рекомендации
            if "healthy" in disease_name.lower():
                result += """
   • Растение здорово
   • Продолжайте регулярный уход
   • Проводите профилактические осмотры
"""
            elif "early_blight" in disease_name.lower():
                result += """
   • Удалите пораженные листья
   • Обработайте фунгицидами
   • Обеспечьте хорошую вентиляцию
"""
            elif "late_blight" in disease_name.lower():
                result += """
   • Срочно удалите пораженные части
   • Используйте медьсодержащие препараты
   • Снизьте влажность
"""
            elif "bacterial" in disease_name.lower():
                result += """
   • Удалите пораженные участки
   • Используйте бактерициды
   • Дезинфицируйте инструменты
"""
            else:
                result += """
   • Изолируйте растение
   • Обратитесь к специалисту
   • Проведите профилактическую обработку
"""

            self.result_text.insert(tk.END, result)

        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"❌ Ошибка:\n\n{str(e)}")


def main():
    root = tk.Tk()
    app = PlantDiagnosisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()