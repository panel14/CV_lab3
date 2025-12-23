import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

model = tf.keras.models.load_model('my_animal_model.keras')

TEST_DATA_DIR = './test_data_ext'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

new_ds = image_dataset_from_directory(
    TEST_DATA_DIR, 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

loss, acc = model.evaluate(new_ds)
print(f"Точность на тестовых данных: {acc:.2%}")