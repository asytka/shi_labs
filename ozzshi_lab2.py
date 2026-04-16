import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model with augmentation
if not os.path.exists('mnist_improved_model.keras'):
    print("\nTraining improved model...")

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=3,
        validation_data=(x_test, y_test),
        verbose=1
    )

    model.save('mnist_improved_model.keras')
    print("Model saved as 'mnist_improved_model.keras'")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    model = keras.models.load_model('mnist_improved_model.keras')
    print("\nModel loaded from disk.")

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy on MNIST: {test_acc:.4f}")

def prepare_photo_image(image_path):
    """
    Enhanced preprocessing for photos of handwritten digits
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Failed to load {image_path}")

    # Convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not find digit in image")

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop digit with padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    digit = img[y:y + h, x:x + w]

    # Resize while maintaining aspect ratio
    target_size = 28
    if digit.shape[0] > digit.shape[1]:
        new_h = target_size
        new_w = int(digit.shape[1] * (target_size / digit.shape[0]))
    else:
        new_w = target_size
        new_h = int(digit.shape[0] * (target_size / digit.shape[1]))

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center the digit
    final_img = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    # Normalize
    final_img = final_img.astype('float32') / 255.0

    # Add dimensions for model input
    final_img = np.expand_dims(final_img, axis=-1)
    final_img = np.expand_dims(final_img, axis=0)

    return final_img

def prepare_simple_image(image_path):
    """
    Simple preprocessing for clear images (scans)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")

    # Invert (black on white -> white on black)
    img = cv2.bitwise_not(img)

    # Resize
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype('float32') / 255.0

    # Add dimensions for model input
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# Process test images
test_folder = "test_images"
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"\nCreated folder '{test_folder}'")
    print("Place your digit photos in this folder and run the program again.")
else:
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(test_folder)
                   if f.lower().endswith(image_extensions)]

    if len(image_files) == 0:
        print(f"\nFolder '{test_folder}' is empty.")
        print("Please add handwritten digit images.")
    else:
        print(f"\nFound {len(image_files)} images in '{test_folder}':")
        print("=" * 70)

        results = []

        for i, fname in enumerate(sorted(image_files), 1):
            path = os.path.join(test_folder, fname)
            print(f"\n{i}. Processing: {fname}")

            try:
                # Try enhanced preprocessing first
                img_prepared = prepare_photo_image(path)

                # Predict
                prediction = model.predict(img_prepared, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                results.append((fname, predicted_digit, confidence))

                print(f"Result: digit {predicted_digit} (confidence: {confidence:.3f})")

                # Display visualization
                plt.figure(figsize=(10, 5))
                
                # Original photo
                original = cv2.imread(path)
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, 1)
                plt.imshow(original_rgb)
                plt.title(f"Original Image", fontsize=12)
                plt.axis('off')
                
                # Processed digit with prediction
                plt.subplot(1, 2, 2)
                processed_display = img_prepared.squeeze()
                plt.imshow(processed_display, cmap='gray')
                plt.title(f"Processed Digit\nPrediction: {predicted_digit} (confidence: {confidence:.2%})", 
                         fontsize=12)
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"ERROR: {e}")
                # Try simple preprocessing as fallback
                try:
                    print("   Trying alternative preprocessing method...")
                    img_prepared = prepare_simple_image(path)
                    prediction = model.predict(img_prepared, verbose=0)
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction)

                    results.append((fname, predicted_digit, confidence))
                    print(f"Result (fallback): digit {predicted_digit} (confidence: {confidence:.3f})")

                    # Display visualization for fallback method
                    plt.figure(figsize=(10, 5))
                    original = cv2.imread(path)
                    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_rgb)
                    plt.title(f"Original Image", fontsize=12)
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    processed_display = img_prepared.squeeze()
                    plt.imshow(processed_display, cmap='gray')
                    plt.title(f"Processed Digit (simple)\nPrediction: {predicted_digit} (confidence: {confidence:.2%})", 
                             fontsize=12)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()

                except Exception as e2:
                    print(f"Fallback method also failed: {e2}")

        # Display summary table
        if results:
            print("\n" + "=" * 70)
            print(f"{'No.':<5} {'Filename':<40} {'Digit':<10} {'Confidence':<10}")
            print("-" * 70)
            for idx, (fname, digit, conf) in enumerate(results, 1):
                print(f"{idx:<5} {fname:<40} {digit:<10} {conf:.3f}")
            print("=" * 70)