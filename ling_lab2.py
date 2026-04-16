import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import librosa
import warnings
warnings.filterwarnings('ignore')

print("\n1. Створення датасету...")

# Класи для розпізнавання
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
print(f"Всього класів: {len(classes)}")

def generate_audio_features(letter, sample_id):
    """Генерація реалістичних аудіо-ознак"""
    
    # Фіксуємо seed для відтворюваності
    letter_hash = abs(hash(letter)) % 10000
    seed_value = (letter_hash + sample_id) % (2**32 - 1)
    np.random.seed(seed_value)
    
    # Розмір ознак: 50 часових кадрів, 13 MFCC коефіцієнтів
    time_steps = 50
    n_mfcc = 13
    
    features = np.zeros((time_steps, n_mfcc))
    
    # Унікальна частота для кожної літери
    if letter.isdigit():
        letter_index = int(letter) + 48
    else:
        letter_index = ord(letter)
    
    base_freq = (letter_index - 48) / 50.0
    
    for t in range(time_steps):
        for f in range(n_mfcc):
            # Головна частота
            main_freq = base_freq * (f + 1) * np.pi
            features[t, f] = np.sin(main_freq * t / time_steps * 2 * np.pi)
            
            # Друга гармоніка
            features[t, f] += 0.3 * np.sin(2 * main_freq * t / time_steps * 2 * np.pi)
            
            # Огинаюча (згасання)
            features[t, f] *= np.exp(-t / 20)
    
    # Додаємо шум
    features += np.random.normal(0, 0.1, features.shape)
    
    # Нормалізація
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)
    
    return features

# Генерація даних
X = []
y = []
samples_per_class = 300

print(f"Генерація {samples_per_class} зразків на клас...")
for i, letter in enumerate(classes):
    for sample_id in range(samples_per_class):
        features = generate_audio_features(letter, sample_id)
        X.append(features)
        y.append(letter)
    
    if (i + 1) % 10 == 0:
        print(f"  Згенеровано {i+1}/{len(classes)} класів")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Датасет створено: {X.shape}")
print(f"   Зразків: {len(X)}")
print(f"   Класів: {len(np.unique(y))}")

# ============================================
# 2. ПІДГОТОВКА ДАНИХ
# ============================================
print("\n2. Підготовка даних...")

# Додаємо канал для CNN
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Кодування міток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Розділення даних
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"   Тренувальна: {X_train.shape[0]} (70%)")
print(f"   Валідаційна: {X_val.shape[0]} (15%)")
print(f"   Тестова: {X_test.shape[0]} (15%)")
print(f"   Розмір ознак: {X_train.shape[1:]}")

# ============================================
# 3. ПОБУДОВА МОДЕЛІ (ВИПРАВЛЕНА)
# ============================================
print("\n3. Побудова нейромережі...")

model = keras.Sequential([
    keras.layers.Input(shape=(50, 13, 1)),
    
    # Перший згортковий блок
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    
    # Другий згортковий блок
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    
    # Третій згортковий блок (БЕЗ MaxPooling, бо розмір вже малий)
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    # Global pooling замість Flatten
    keras.layers.GlobalAveragePooling2D(),
    
    # Повнозв'язні шари
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    
    # Вихідний шар
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 4. НАВЧАННЯ
# ============================================
print("\n4. Навчання моделі...")

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 5. ОЦІНКА
# ============================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТИ ОЦІНКИ")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Точність на тестових даних: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Передбачення
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================
# 6. ФУНКЦІЯ РОЗПІЗНАВАННЯ
# ============================================
def recognize_audio(audio_features):
    """Розпізнавання мовлення з аудіо-ознак"""
    if len(audio_features.shape) == 2:
        audio_features = audio_features.reshape(1, 50, 13, 1)
    elif len(audio_features.shape) == 3:
        audio_features = audio_features.reshape(1, audio_features.shape[0], audio_features.shape[1], 1)
    
    pred_probs = model.predict(audio_features, verbose=0)
    pred_class = np.argmax(pred_probs[0])
    confidence = float(pred_probs[0][pred_class])
    
    return label_encoder.inverse_transform([pred_class])[0], confidence

# ============================================
# 7. ТЕСТУВАННЯ
# ============================================
print("\n" + "="*60)
print("ТЕСТУВАННЯ ІНДИВІДУАЛЬНИХ ЗРАЗКІВ")
print("="*60)

test_indices = np.random.choice(len(X_test), min(25, len(X_test)), replace=False)
correct = 0

for i, idx in enumerate(test_indices):
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    predicted, confidence = recognize_audio(X_test[idx])
    
    status = "✓" if predicted == true_label else "✗"
    if predicted == true_label:
        correct += 1
    
    print(f"{status} {i+1:2d}: '{predicted}' (true: '{true_label}') - {confidence:.3f}")

print(f"\n✅ Точність: {correct}/{len(test_indices)} = {correct/len(test_indices)*100:.1f}%")

# ============================================
# 8. РОЗПІЗНАВАННЯ БОРТОВИХ НОМЕРІВ
# ============================================
print("\n" + "="*60)
print("РОЗПІЗНАВАННЯ БОРТОВИХ НОМЕРІВ")
print("="*60)

test_flights = [
    ['A', 'B', '1', '2', '3'],
    ['F', 'L', '3', '5', '2'],
    ['N', '7', '3', 'A'],
    ['C', 'D', '5', '6'],
    ['Z', '9', '8', '8']
]

print("\n✈️ Результати:")

# Створюємо словник для швидкого пошуку зразків
test_samples = {}
for i, label in enumerate(label_encoder.inverse_transform(y_test)):
    if label not in test_samples:
        test_samples[label] = X_test[i]

for flight in test_flights:
    recognized = []
    confidences = []
    
    for char in flight:
        if char in test_samples:
            pred, conf = recognize_audio(test_samples[char])
            recognized.append(pred)
            confidences.append(conf)
        else:
            # Якщо немає в тестовій вибірці, шукаємо в тренувальній
            train_indices = np.where(label_encoder.inverse_transform(y_train) == char)[0]
            if len(train_indices) > 0:
                pred, conf = recognize_audio(X_train[train_indices[0]])
                recognized.append(pred)
                confidences.append(conf)
    
    if len(recognized) == len(flight):
        recognized_str = ''.join(recognized)
        expected_str = ''.join(flight)
        avg_conf = np.mean(confidences)
        status = "✓" if recognized_str == expected_str else "✗"
        print(f"   {status} '{expected_str}' → '{recognized_str}' (conf: {avg_conf:.3f})")
    else:
        print(f"   ✗ Не вдалося розпізнати '{''.join(flight)}'")

# ============================================
# 9. ВІЗУАЛІЗАЦІЯ
# ============================================
print("\n9. Генерація графіків...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Точність
axes[0, 0].plot(history.history['accuracy'], label='Тренування', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Валідація', linewidth=2)
axes[0, 0].set_xlabel('Епоха')
axes[0, 0].set_ylabel('Точність')
axes[0, 0].set_title('Точність моделі')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Втрати
axes[0, 1].plot(history.history['loss'], label='Тренування', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Валідація', linewidth=2)
axes[0, 1].set_xlabel('Епоха')
axes[0, 1].set_ylabel('Втрати')
axes[0, 1].set_title('Втрати моделі')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Матриця помилок (перші 12 класів)
y_true = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
first_12 = classes[:12]
cm = confusion_matrix(y_true, y_pred_labels, labels=first_12)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=first_12, yticklabels=first_12, ax=axes[0, 2])
axes[0, 2].set_title('Матриця помилок')
axes[0, 2].set_xlabel('Передбачено')
axes[0, 2].set_ylabel('Істинне')

# Точність по класах
class_acc = {}
for cls in first_12:
    mask = y_true == cls
    if mask.sum() > 0:
        class_acc[cls] = (y_pred_labels[mask] == y_true[mask]).mean()

axes[1, 0].bar(range(len(class_acc)), list(class_acc.values()))
axes[1, 0].set_xticks(range(len(class_acc)))
axes[1, 0].set_xticklabels(list(class_acc.keys()), rotation=45)
axes[1, 0].set_xlabel('Клас')
axes[1, 0].set_ylabel('Точність')
axes[1, 0].set_title('Точність по класах')
axes[1, 0].set_ylim([0, 1])

# Розподіл впевненості
confidences = np.max(y_pred_probs, axis=1)
axes[1, 1].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Впевненість')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].set_title(f'Розподіл впевненості\nСереднє: {np.mean(confidences):.3f}')

# Приклад ознак
axes[1, 2].imshow(X_test[0, :, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
axes[1, 2].set_xlabel('Час')
axes[1, 2].set_ylabel('Частота')
axes[1, 2].set_title('Спектрограма аудіо')

plt.tight_layout()
plt.show()

model.save('speech_recognition_model.keras')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\n💾 Модель збережено: speech_recognition_model.keras")
print("💾 Кодувальник міток: label_encoder.pkl")
