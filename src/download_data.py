import os
import urllib.request
import tarfile
import pandas as pd

def download_and_extract_imdb(data_dir='data'):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extracted_path = os.path.join(data_dir, "aclImdb")

    os.makedirs(data_dir, exist_ok=True)  # создаем папку если нет

    if not os.path.exists(filename):
        print("Скачивание датасета IMDB...")
        urllib.request.urlretrieve(url, filename)
        print("Датасет скачан.")
    else:
        print("Файл уже существует.")

    if not os.path.exists(extracted_path):
        print("Распаковка архива...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Распаковка завершена.")
    else:
        print("Архив уже распакован.")

    print("Содержимое папки aclImdb:")
    print(os.listdir(extracted_path))

    return extracted_path  


# Чтение отзывов из папки (train/pos и train/neg или test/pos и test/neg)
def load_reviews(data_dir):

    data = []
    for sentiment in ['pos', 'neg']:
        folder_path = os.path.join(data_dir, sentiment)
        label = 1 if sentiment == 'pos' else 0

        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена.")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                rating = int(filename.split('_')[1].split('.')[0])  # считываем рейтинг из названия файла
                with open(file_path, 'r', encoding='utf-8') as file:
                    review_text = file.read()
                data.append({'review': review_text, 'rating': rating, 'label': label})

    return pd.DataFrame(data)
