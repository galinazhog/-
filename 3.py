import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import random
# Ячейка 2.2 - Формирование последовательностей пользователей
print("Формирование последовательностей пользователей...")

# Загружаем данные
ratings_df = pd.read_csv('./data/processed/ratings_processed.csv')
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])

# Загружаем словари
with open('./data/processed/movie2idx.pkl', 'rb') as f:
    movie2idx = pickle.load(f)

# Разделяем положительные и отрицательные взаимодействия
pos_ratings = ratings_df[ratings_df['rating'] >= 4.0]
neg_ratings = ratings_df[ratings_df['rating'] <= 2.0]
neutral_ratings = ratings_df[(ratings_df['rating'] > 2.0) & (ratings_df['rating'] < 4.0)]

print(f"Положительных оценок (>=4): {len(pos_ratings):,}")
print(f"Отрицательных оценок (<=2): {len(neg_ratings):,}")
print(f"Нейтральных оценок (2<rating<4): {len(neutral_ratings):,}")

# Создание последовательностей
user_sequences = defaultdict(lambda: {'positive': [], 'negative': [], 'timestamps': []})

# Добавляем положительные последовательности (с сортировкой по времени)
for user_id, group in tqdm(pos_ratings.groupby('userId'), desc="Обработка положительных оценок"):
    # Сортируем по timestamp
    sorted_group = group.sort_values('timestamp')
    seq = sorted_group['movieId'].tolist()
    timestamps = sorted_group['timestamp'].tolist()

    # Фильтруем только те фильмы, которые есть в словаре
    valid_seq = []
    valid_timestamps = []
    for m, t in zip(seq, timestamps):
        if m in movie2idx:
            valid_seq.append(movie2idx[m])
            valid_timestamps.append(t)

    if len(valid_seq) >= 3:  # Минимум 3 фильма для обучения
        user_sequences[user_id]['positive'] = valid_seq
        user_sequences[user_id]['timestamps'] = valid_timestamps

# Добавляем отрицательные фильмы
for user_id, group in tqdm(neg_ratings.groupby('userId'), desc="Обработка отрицательных оценок"):
    if user_id in user_sequences:
        neg_movies = []
        for m in group['movieId'].tolist():
            if m in movie2idx:
                neg_movies.append(movie2idx[m])
        if neg_movies:
            user_sequences[user_id]['negative'] = neg_movies

# Сохраняем последовательности
with open('./data/processed/user_sequences.pkl', 'wb') as f:
    pickle.dump(dict(user_sequences), f)

print(f"\n Создано последовательностей для {len(user_sequences)} пользователей")

# Демонстрация примера
sample_user = list(user_sequences.keys())[0]
sample_data = user_sequences[sample_user]
print(f"\n Пример для пользователя {sample_user}:")
print(f"  Количество фильмов в истории: {len(sample_data['positive'])}")
print(f"  Количество отрицательных фильмов: {len(sample_data.get('negative', []))}")
if sample_data.get('timestamps'):
    print(f"  Период: {sample_data['timestamps'][0].date()} - {sample_data['timestamps'][-1].date()}")

