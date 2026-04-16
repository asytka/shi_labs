import matplotlib.pyplot as plt
import requests
import gzip
import numpy as np
import os

data_dir = "../_data"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
mnist_dataset = {}

def one_hot_encoding(labels, dimension=10):
    """One-hot encoding для міток"""
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    return one_hot_labels.astype(np.float64)

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status()
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)

for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

print(f"   Навчальні зображення: {x_train.shape}")
print(f"   Навчальні мітки: {y_train.shape}")
print(f"   Тестові зображення: {x_test.shape}")
print(f"   Тестові мітки: {y_test.shape}")

mnist_image = x_train[59999, :].reshape(28, 28)
fig, ax = plt.subplots()
ax.imshow(mnist_image, cmap="gray")
ax.set_title("Приклад зображення з MNIST")
ax.axis('off')
plt.show()

num_examples = 5
seed = 147197952744
rng = np.random.default_rng(seed)

fig, axes = plt.subplots(1, num_examples, figsize=(12, 3))
for i, (sample, ax) in enumerate(zip(rng.choice(x_train, size=num_examples, replace=False), axes)):
    ax.imshow(sample.reshape(28, 28), cmap="gray")
    ax.axis('off')

plt.suptitle("5 випадкових зображень з MNIST")
plt.tight_layout()
plt.show()

training_sample = 10000 
test_sample = 2000      

training_images = x_train[:training_sample] / 255.0
test_images = x_test[:test_sample] / 255.0

training_labels = one_hot_encoding(y_train[:training_sample])
test_labels = one_hot_encoding(y_test[:test_sample])

print(f"\n Підготовка даних:")
print(f"   Навчальна вибірка: {len(training_images)} зображень")
print(f"   Тестова вибірка: {len(test_images)} зображень")
print(f"   Розмірність зображення: 28x28 = 784 пікселі")
print(f"   Кількість класів: 10 (цифри 0-9)")


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax function for better probability distribution"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

seed = 884736743
rng = np.random.default_rng(seed)

learning_rate = 0.01  
epochs = 30            
hidden_size = 128      
pixels_per_image = 784
num_labels = 10

weights_1 = rng.normal(0, np.sqrt(2.0 / pixels_per_image), (pixels_per_image, hidden_size))
weights_2 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, num_labels))

print(f"   Вхідний шар: {pixels_per_image} нейронів")
print(f"   Прихований шар: {hidden_size} нейронів")
print(f"   Вихідний шар: {num_labels} нейронів")
print(f"   Функція активації: ReLU")
print(f"   Learning rate: {learning_rate}")
print(f"   Епох: {epochs}")

store_training_loss = []
store_training_accuracy = []
store_test_loss = []
store_test_accuracy = []

for epoch in range(epochs):
    training_loss = 0.0
    training_correct = 0
    
    for i in range(len(training_images)):
        layer_0 = training_images[i]
        layer_1 = relu(np.dot(layer_0, weights_1))
        
        dropout_rate = 0.5
        dropout_mask = (rng.random(layer_1.shape) > dropout_rate).astype(float)
        layer_1_dropped = layer_1 * dropout_mask / (1 - dropout_rate)
        
        layer_2 = np.dot(layer_1_dropped, weights_2)
        
        layer_2_softmax = softmax(layer_2.reshape(1, -1)).flatten()
        
        loss = -np.log(layer_2_softmax[np.argmax(training_labels[i])] + 1e-8)
        training_loss += loss
        
        if np.argmax(layer_2_softmax) == np.argmax(training_labels[i]):
            training_correct += 1
        
        layer_2_delta = layer_2_softmax - training_labels[i]
        
        layer_1_delta = np.dot(layer_2_delta, weights_2.T) * relu_derivative(layer_1)
        
        layer_1_delta *= dropout_mask
        
        weights_2 -= learning_rate * np.outer(layer_1_dropped, layer_2_delta)
        weights_1 -= learning_rate * np.outer(layer_0, layer_1_delta)
    
    test_layer_1 = relu(np.dot(test_images, weights_1))
    test_layer_2 = softmax(np.dot(test_layer_1, weights_2))
    
    test_loss = -np.mean(np.log(test_layer_2[np.arange(len(test_labels)), np.argmax(test_labels, axis=1)] + 1e-8))
    
    test_correct = np.sum(np.argmax(test_layer_2, axis=1) == np.argmax(test_labels, axis=1))
    test_accuracy = test_correct / len(test_images)
    
    train_accuracy = training_correct / len(training_images)
    train_loss = training_loss / len(training_images)
    
    store_training_loss.append(train_loss)
    store_training_accuracy.append(train_accuracy)
    store_test_loss.append(test_loss)
    store_test_accuracy.append(test_accuracy)
    
    if (epoch + 1) % 5 == 0:
        print(f"Епоха {epoch+1:2d}/{epochs}: "
              f"Train Acc = {train_accuracy:.4f} ({train_accuracy*100:.2f}%), "
              f"Test Acc = {test_accuracy:.4f} ({test_accuracy*100:.2f}%), "
              f"Test Loss = {test_loss:.4f}")

epoch_range = np.arange(1, epochs + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(epoch_range, store_training_accuracy, 'b-', label='Навчальна', linewidth=2)
axes[0, 0].plot(epoch_range, store_test_accuracy, 'r-', label='Тестова', linewidth=2)
axes[0, 0].set_title('Точність моделі', fontsize=14)
axes[0, 0].set_xlabel('Епоха')
axes[0, 0].set_ylabel('Точність')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

axes[0, 1].plot(epoch_range, store_training_loss, 'b-', label='Навчальна', linewidth=2)
axes[0, 1].plot(epoch_range, store_test_loss, 'r-', label='Тестова', linewidth=2)
axes[0, 1].set_title('Функція втрат (Cross-Entropy)', fontsize=14)
axes[0, 1].set_xlabel('Епоха')
axes[0, 1].set_ylabel('Втрати')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

test_predictions = np.argmax(test_layer_2, axis=1)
test_true = np.argmax(test_labels, axis=1)

correct_idx = np.where(test_predictions == test_true)[0]
wrong_idx = np.where(test_predictions != test_true)[0]

for i, idx in enumerate(correct_idx[:4]):
    ax = axes[1, 0] if i < 2 else axes[1, 1]
    if i == 2:
        ax = axes[1, 1]
    if i < 2:
        img_idx = idx
        ax.imshow(x_test[img_idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Передбачено: {test_predictions[img_idx]}, Справжнє: {test_true[img_idx]}', fontsize=10)
        ax.axis('off')

axes[1, 0].set_title('Правильні передбачення (приклади)', fontsize=12)

for i, idx in enumerate(wrong_idx[:4]):
    if i < 2:
        ax = axes[1, 1] if i == 0 else None
        if i == 0:
            img_idx = idx
            axes[1, 1].imshow(x_test[img_idx].reshape(28, 28), cmap='gray')
            axes[1, 1].set_title(f'Передбачено: {test_predictions[img_idx]}, Справжнє: {test_true[img_idx]}', fontsize=10)
            axes[1, 1].axis('off')

axes[1, 1].set_title('Неправильні передбачення (приклади)', fontsize=12)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

correct_count = 0
for i in range(16):
    idx = np.random.randint(0, len(test_images))
    img = test_images[idx].reshape(28, 28)
    true_label = test_true[idx]
    pred_label = test_predictions[idx]
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {true_label} | Pred: {pred_label}', fontsize=10)
    axes[i].axis('off')
    
    if true_label == pred_label:
        axes[i].title.set_color('green')
        correct_count += 1
    else:
        axes[i].title.set_color('red')

plt.suptitle(f'Результати розпізнавання цифр - {correct_count}/16 правильно ({correct_count/16*100:.0f}%)', 
             fontsize=14)
plt.tight_layout()
plt.show()


final_train_accuracy = store_training_accuracy[-1]
final_test_accuracy = store_test_accuracy[-1]

