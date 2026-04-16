import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_emnist_images(filename):
    """Load EMNIST images from ubyte file"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data

def load_emnist_labels(filename):
    """Load EMNIST labels from ubyte file"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

data_dir = 'gzip'

train_images_path = os.path.join(data_dir, 'emnist-balanced-train-images-idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'emnist-balanced-train-labels-idx1-ubyte')
test_images_path = os.path.join(data_dir, 'emnist-balanced-test-images-idx3-ubyte')
test_labels_path = os.path.join(data_dir, 'emnist-balanced-test-labels-idx1-ubyte')

print("\n1. Loading EMNIST Balanced dataset...")
X_train_full = load_emnist_images(train_images_path)
y_train_full = load_emnist_labels(train_labels_path)
X_test_original = load_emnist_images(test_images_path)
y_test_original = load_emnist_labels(test_labels_path)

print(f"   Training data shape: {X_train_full.shape}")
print(f"   Number of classes: {len(np.unique(y_train_full))}")

# Fix EMNIST orientation
print("\n2. Preprocessing images...")
X_train_full = np.rot90(X_train_full, k=1, axes=(1,2))
X_train_full = np.fliplr(X_train_full.reshape(-1, 28, 28)).reshape(-1, 28, 28)
X_test_original = np.rot90(X_test_original, k=1, axes=(1,2))
X_test_original = np.fliplr(X_test_original.reshape(-1, 28, 28)).reshape(-1, 28, 28)

# Normalize
X_train_full = X_train_full.astype('float32') / 255.0
X_test_original = X_test_original.astype('float32') / 255.0

print("\n3. Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"   Training: {X_train.shape[0]} (70%)")
print(f"   Validation: {X_val.shape[0]} (15%)")
print(f"   Test: {X_test.shape[0]} (15%)")

print("\n4. Building Neural Network...")
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(47, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n5. Training the model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=128,
                    verbose=1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

def class_to_char(class_id):
    if 0 <= class_id <= 9:
        return str(class_id)
    elif 10 <= class_id <= 35:
        return chr(class_id - 10 + ord('A'))
    else:
        return chr(class_id - 36 + ord('a'))

# Get predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# ============================================
# 6. VISUALIZATIONS - CORRECTED VERSION
# ============================================
print("\n6. Generating visualizations...")

# Create figure with proper spacing
fig = plt.figure(figsize=(18, 12))

# Plot 1: Training History - Accuracy
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
ax1.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - Loss
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['loss'], 'b-', label='Training', linewidth=2)
ax2.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sample Predictions (16 images in a grid)
ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
# Create a separate figure for the grid of predictions
fig2 = plt.figure(figsize=(10, 10))
for i in range(16):
    ax = fig2.add_subplot(4, 4, i+1)
    ax.imshow(X_test[i], cmap='gray')
    pred_class = y_pred_classes[i]
    true_class = y_test[i]
    pred_char = class_to_char(pred_class)
    true_char = class_to_char(true_class)
    color = 'green' if pred_class == true_class else 'red'
    ax.set_title(f'Pred: {pred_char}\nTrue: {true_char}', fontsize=8, color=color)
    ax.axis('off')
fig2.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.show()

# Plot 4: Confusion Matrix for digits
ax4 = plt.subplot(2, 3, 4)
y_test_digits = y_test[y_test < 10]
y_pred_digits = y_pred_classes[y_test < 10]
cm_digits = confusion_matrix(y_test_digits, y_pred_digits)
sns.heatmap(cm_digits, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)], ax=ax4)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('True')
ax4.set_title('Confusion Matrix - Digits (0-9)')

# Plot 5: Per-class accuracy
ax5 = plt.subplot(2, 3, 5)
class_accuracy = {}
for i in range(47):
    mask = (y_test == i)
    if np.sum(mask) > 0:
        acc = np.sum(y_pred_classes[mask] == y_test[mask]) / np.sum(mask)
        class_accuracy[class_to_char(i)] = acc

# Show first 20 classes
chars = list(class_accuracy.keys())[:20]
accs = list(class_accuracy.values())[:20]
ax5.bar(chars, accs)
ax5.set_xlabel('Character')
ax5.set_ylabel('Accuracy')
ax5.set_title('Per-Class Accuracy (First 20 classes)')
ax5.set_xticklabels(chars, rotation=45)
ax5.set_ylim([0, 1])


correct = 0
for i in range(25):
    pred_char = class_to_char(y_pred_classes[i])
    true_char = class_to_char(y_test[i])
    confidence = np.max(y_pred[i])
    status = "✓" if y_pred_classes[i] == y_test[i] else "✗"
    if y_pred_classes[i] == y_test[i]:
        correct += 1
    print(f"{status} Sample {i+1:2d}: Predicted = '{pred_char}', True = '{true_char}', Confidence = {confidence:.4f}")

print(f"\n✅ Accuracy on 25 samples: {correct}/25 = {correct/25*100:.1f}%")

model.save('text_recognition_model.h5')
print("\n💾 Model saved as 'text_recognition_model.h5'")
