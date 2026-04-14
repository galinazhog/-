import os
import pandas as pd
import numpy as np

# Ячейка 2.1 - Фильтрация пользователей и создание словарей
import pickle
from collections import defaultdict
from tqdm import tqdm
import random

print("Фильтрация пользователей...")

# Проверяем наличие файла
if not os.path.exists('./data/processed/ratings_processed.csv'):
    print(" Файл ratings_processed.csv не найден!")
    print("Пожалуйста, сначала выполните Этап 1")
else:
    # Загружаем обработанные данные
    ratings_df = pd.read_csv('./data/processed/ratings_processed.csv')
    # Конвертируем timestamp обратно в datetime при загрузке
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])

    print(f"Загружено {len(ratings_df)} записей")

    # Фильтруем пользователей с малым количеством оценок
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 10].index
    ratings_filtered = ratings_df[ratings_df['userId'].isin(valid_users)]
    print(f"Пользователей после фильтрации (мин. 10 оценок): {ratings_filtered['userId'].nunique()}")

    # Создание словарей для фильмов
    movie_ids = ratings_filtered['movieId'].unique()
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3
    NUM_SPECIAL_TOKENS = 4

    movie2idx = {movie_id: idx + NUM_SPECIAL_TOKENS for idx, movie_id in enumerate(movie_ids)}
    idx2movie = {idx: movie_id for movie_id, idx in movie2idx.items()}
    num_movies = len(movie_ids) + NUM_SPECIAL_TOKENS

    print(f"Размер словаря фильмов: {len(movie_ids)}")
    print(f"Размер словаря со спецтокенами: {num_movies}")

    # Сохраняем словари
    with open('./data/processed/movie2idx.pkl', 'wb') as f:
        pickle.dump(movie2idx, f)
    with open('./data/processed/idx2movie.pkl', 'wb') as f:
        pickle.dump(idx2movie, f)

    # Сохраняем параметры
    config = {
        'num_movies': num_movies,
        'pad_token': PAD_TOKEN,
        'bos_token': BOS_TOKEN,
        'eos_token': EOS_TOKEN,
        'unk_token': UNK_TOKEN,
        'num_special_tokens': NUM_SPECIAL_TOKENS
    }
    with open('./data/processed/config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print(" Словари сохранены")

