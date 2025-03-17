# Модель для предсказания тональности и рейтинга отзывов (IMDB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Diank/IMDB_sentiment_rating/blob/main/notebooks/demo_pipeline.ipynb)


## Цель проекта:
- Предсказать тональность отзыва о фильме (положительный/отрицательный)
- Предсказать рейтинг отзыва: 
	- рейтинг в диапазоне [1; 4] для отрицательного отзыва 
	- рейтинг в диапазоне [7; 10] для положительного отзыва

## Описание модели
Модель построена на **Bidirectional LSTM**, которая одновременно решает две задачи:
- Классификация тональности (бинарная классификация)
- Восстановление рейтинга (регрессия), с учетом предсказанной тональности

## Результаты на тестовых данных:
- Accuracy (классификация тональности): `0.8734`
- MAE (средняя абсолютная ошибка рейтинга): `1.9532`

## Используемые технологии:
- Python 3.x
- PyTorch
- NLTK (токенизация)
- Pandas / NumPy
- Matplotlib / Seaborn (визуализация)
- EarlyStopping, Scheduler
- Google Colab (для обучения и демонстрации)

## Возможности:
- Одновременное предсказание тональности и рейтинга
- Dynamic Padding для экономии памяти
- Early Stopping для предотвращения переобучения
- Learning Rate Scheduler
- Поддержка кастомного словаря с ограничением по размеру

## Структура проекта:
```
IMDB-sentiment-rating/
│
├── src/
│   ├── collate.py - выравнивание бача
│   ├── data.py — подготовка данных
│   ├── download_data.py
│   ├── model.py — модель
│   ├── train.py — обучение
│   ├── utils.py — графики
│
├── notebooks/
│   ├── demo_pipeline.ipynb — демонстрация пайплайна
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Как запустить проект

### 1. Клонировать репозиторий
```
git clone https://github.com/Diank/IMDB_sentiment_rating.git
cd IMDB_sentiment_rating
```

### 2. Установить зависимости (если запускаете локально):
```
pip install -r requirements.txt
```

### 3. Запустить демонстрационный ноутбук
Откройте demo_pipeline.ipynb:
- Рекомендуется запускать в Google Colab
- Можно также запустить локально через Jupyter Notebook

### 4. Что делает ноутбук:
- Скачивает и распаковывает датасет IMDB (автоматически)
- Предобрабатывает текстовые данные
- Обучает модель
- Строит графики метрик (loss, accuracy, MAE)
- Оценивает финальную модель на тестовой выборке
- Предсказывает тональность и рейтинг отзыва для одного примера

## Возможные улучшения
- Подобрать более оптимальные гиперпараметры
- Изменить архитектуру модели
- Использовать предобученные эмбеддинги (GloVe, FastText) или большие языковые модели (например, BERT)
- Добавить более сложный pipeline обработки текста
















