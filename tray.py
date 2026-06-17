from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = generator.flow_from_directory(
    "PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = generator.flow_from_directory(
    "PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Загружаем базовую модель
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Замораживаем её слои
base_model.trainable = False

# Добавляем свои слои
x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

model = Model(base_model.input, predictions)

# Компилируем модель
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Показываем структуру модели
model.summary()
print(train_data.class_indices)


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

model.save("plant_model.keras")

print("Модель успешно сохранена!")

