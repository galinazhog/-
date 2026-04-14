import os
import pandas as pd
import numpy as np

# Создаем необходимые директории
os.makedirs('./data/processed', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)

print("Создание директорий завершено")

# Загрузка данных
print("Загрузка данных...")

# Проверяем наличие файлов
if not os.path.exists('./ml-20m/rating.csv'):
    print("Файлы не найдены! Пожалуйста, скачайте датасет с Kaggle и распакуйте в папку './ml-20m/'")
    print("Ссылка: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset")
else:
    # Загружаем данные
    ratings_df = pd.read_csv('./ml-20m/rating.csv')
    movies_df = pd.read_csv('./ml-20m/movie.csv')

    # Проверяем, есть ли файл tag.csv
    tags_df = None
    if os.path.exists('./ml-20m/tag.csv'):
        tags_df = pd.read_csv('./ml-20m/tag.csv')

    print(f"Размер ratings.csv: {ratings_df.shape}")
    print(f"Размер movies.csv: {movies_df.shape}")
    if tags_df is not None:
        print(f"Размер tag.csv: {tags_df.shape}")

    # Проверяем тип данных timestamp
    print(f"\nТип данных timestamp: {ratings_df['timestamp'].dtype}")
    print(f"Пример значения timestamp: {ratings_df['timestamp'].iloc[0]}")

    # Конвертируем timestamp в datetime
    # Если timestamp - это число (Unix timestamp)
    if pd.api.types.is_numeric_dtype(ratings_df['timestamp']):
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        print("Timestamp сконвертирован из Unix timestamp")
    else:
        # Если timestamp - это строка с датой
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
        print("Timestamp сконвертирован из строки с датой")

    # Если есть tags, конвертируем и там
    if tags_df is not None:
        if pd.api.types.is_numeric_dtype(tags_df['timestamp']):
            tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'], unit='s')
        else:
            tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'])

    print(f"\nДиапазон дат: {ratings_df['timestamp'].min()} - {ratings_df['timestamp'].max()}")

    # Сохраняем обработанные данные в правильную директорию
    ratings_df.to_csv('./data/processed/ratings_processed.csv', index=False)
    movies_df.to_csv('./data/processed/movies_processed.csv', index=False)
    if tags_df is not None:
        tags_df.to_csv('./data/processed/tags_processed.csv', index=False)

    print("\nДанные успешно загружены и сохранены в './data/processed/'")

    # Выводим базовую статистику
    print(f"\nСтатистика по данным:")
    print(f"  Уникальных пользователей: {ratings_df['userId'].nunique()}")
    print(f"  Уникальных фильмов: {ratings_df['movieId'].nunique()}")
    print(f"  Диапазон рейтингов: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    print(f"  Средний рейтинг: {ratings_df['rating'].mean():.2f}")

