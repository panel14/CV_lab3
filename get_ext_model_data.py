import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
TEST_DIR = './test_data'

model = tf.keras.models.load_model('my_animal_model.keras')

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False 
)

class_names = test_ds.class_names
print(f"Классы: {class_names}")

y_pred_raw = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred_raw, axis=1)

y_true = np.concatenate([y for x, y in test_ds], axis=0)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Предсказание модели')
plt.ylabel('На самом деле (Истина)')
plt.title('Confusion Matrix')
plt.show()

print("\n--- Детальный отчет по классам ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names))