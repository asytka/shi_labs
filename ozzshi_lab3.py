"""
Лабораторна робота №3: Побудова панорамних зображень
ВИПРАВЛЕНА ВЕРСІЯ - без артефактів та чорних областей
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')


def crop_black_borders(image):
    """
    Обрізає чорні краї зображення
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Знаходимо всі нечорні пікселі
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


def blend_images(img1, img2, mask):
    """
    Плавне накладання двох зображень за допомогою маски
    """
    # Створюємо градієнтну маску для плавного переходу
    gradient = np.zeros_like(mask, dtype=np.float32)
    for i in range(gradient.shape[1]):
        alpha = i / gradient.shape[1]
        gradient[:, i] = alpha

    # Накладаємо з плавним переходом
    result = np.zeros_like(img1, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = img1[:, :, c] * (1 - gradient) + img2[:, :, c] * gradient

    # Там де немає пікселів в img1, беремо з img2
    mask1 = (img1 == 0).all(axis=2)
    for c in range(3):
        result[mask1, c] = img2[mask1, c]

    return result.astype(np.uint8)


def find_homography_and_stitch(img_left, img_right):
    """
    Знаходить гомографію і склеює ліве з правим зображенням
    """
    # Зменшуємо для швидкості
    scale = 0.5
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    img_left_small = cv2.resize(img_left, (int(w_left * scale), int(h_left * scale)))
    img_right_small = cv2.resize(img_right, (int(w_right * scale), int(h_right * scale)))

    # Знаходимо ключові точки
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left_small, None)
    kp2, des2 = sift.detectAndCompute(img_right_small, None)

    if des1 is None or des2 is None:
        return None

    # Співставляємо
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Тест Лоу
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        print(f"   Замало співпадінь: {len(good_matches)}")
        return None

    # Отримуємо координати
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Масштабуємо координати назад
    src_pts = src_pts / scale
    dst_pts = dst_pts / scale

    # Знаходимо гомографію
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None

    # Склеюємо
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # Визначаємо розмір панорами
    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)

    all_corners = np.vstack((transformed.reshape(-1, 2),
                              np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    shift = [-xmin, -ymin]
    H_shift = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    H_final = H_shift @ H

    # Трансформуємо праве зображення
    panorama = cv2.warpPerspective(img_right, H_final, (xmax - xmin, ymax - ymin))

    # Створюємо маску для правого зображення (де є пікселі)
    mask_right = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8), H_final,
                                      (xmax - xmin, ymax - ymin))

    # Вставляємо ліве зображення тільки там, де немає правого
    for c in range(3):
        panorama[shift[1]:shift[1] + h1, shift[0]:shift[0] + w1, c] = \
            np.where(mask_right[shift[1]:shift[1] + h1, shift[0]:shift[0] + w1] == 0,
                     img_left[:, :, c],
                     panorama[shift[1]:shift[1] + h1, shift[0]:shift[0] + w1, c])

    return panorama


def stitch_sequence(images):
    """
    Склеює послідовність зображень у правильному порядку
    """
    if len(images) == 0:
        return None

    if len(images) == 1:
        return images[0]

    # Починаємо з першого зображення
    result = images[0]

    for i in range(1, len(images)):
        print(f"\n   Склеювання зображення {i+1} до результату...")
        new_result = find_homography_and_stitch(result, images[i])

        if new_result is not None:
            result = new_result
            print(f"Успішно склеєно! Розмір: {result.shape[1]}x{result.shape[0]}")
        else:
            print(f"Не вдалося склеїти, пропускаємо зображення {i+1}")

    return result


def show_keypoints_and_matches(img1, img2):
    """
    Показує ключові точки та співпадіння між двома зображеннями
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("   Не знайдено дескрипторів")
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 8))
    plt.imshow(img_rgb)
    plt.title(f"Співпадіння: {len(good_matches)} точок", fontsize=12)
    plt.axis('off')
    plt.show()

    return len(good_matches)


def load_images(folder_path):
    """
    Завантажує зображення з папки
    """
    images = []
    filenames = []

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    files.sort()

    print("\nЗавантажені файли:")
    for filename in files:
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
            print(f"{filename} ({img.shape[1]}x{img.shape[0]})")
        else:
            print(f"Не вдалося завантажити {filename}")

    return images, filenames


def main():
    panorama_folder = "panorama_images"

    if not os.path.exists(panorama_folder):
        os.makedirs(panorama_folder)
        print(f"\nСтворено папку '{panorama_folder}'")
        print("\nІНСТРУКЦІЯ:")
        print("   1. Помістіть 2-5 фотографій в папку 'panorama_images'")
        print("   2. Фото мають йти ПО ПОРЯДКУ (зліва направо)")
        print("   3. Назвіть файли: 01.jpg, 02.jpg, 03.jpg...")
        print("   4. Запустіть програму знову")
        return

    # Завантажуємо
    images, filenames = load_images(panorama_folder)

    if len(images) < 2:
        print(f"\nЗнайдено лише {len(images)} зображень. Потрібно мінімум 2.")
        return

    print(f"\n✅ Завантажено {len(images)} зображень")

    # Перевіряємо співпадіння між сусідніми фото
    for i in range(len(images) - 1):
        print(f"\nМіж {filenames[i]} та {filenames[i+1]}:")
        n_matches = show_keypoints_and_matches(images[i], images[i+1])

        if n_matches < 20:
            print(f"Замало співпадінь ({n_matches})")
            print("Порада: збільште перекриття між фото")

    # Створюємо панораму

    panorama = stitch_sequence(images)

    if panorama is not None:
        # Обрізаємо чорні краї
        print("\nОбрізка чорних країв...")
        panorama = crop_black_borders(panorama)

        # Зберігаємо
        output_path = "panorama_result.jpg"
        cv2.imwrite(output_path, panorama)
        print(f"Збережено: {output_path}")

        # Показуємо
        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(16, 8))
        plt.imshow(panorama_rgb)
        plt.title("Панорама", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("panorama_display.png", dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nРозмір панорами: {panorama.shape[1]} x {panorama.shape[0]} пікселів")
        print(f"\nСтатистика: {len(images)} фото → {panorama.shape[1]}x{panorama.shape[0]}")

    else:
        print("\nНЕ ВДАЛОСЯ СТВОРИТИ ПАНОРАМУ")
        print("\nПОРАДИ:")
        print("   1. Зробіть нові фото з БІЛЬШИМ перекриттям (40-50%)")
        print("   2. Фотографуйте контрастні об'єкти")
        print("   3. Почніть з 2 фото замість 3-х")


if __name__ == "__main__":
    main()