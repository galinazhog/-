# 7_baselines_optimized.py - Оптимизированные baseline модели
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
import gc
import sys

print("="*60)
print("BASELINE МОДЕЛИ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
print("="*60)

# ============================================
# 1. ЗАГРУЗКА ТОЛЬКО НЕОБХОДИМЫХ ДАННЫХ
# ============================================
print("\n1. Загрузка минимально необходимых данных...")

# Загружаем только конфигурацию и словари (маленькие файлы)
with open('./data/processed/config.pkl', 'rb') as f:
    config_data = pickle.load(f)

with open('./data/processed/movie2idx.pkl', 'rb') as f:
    movie2idx = pickle.load(f)

with open('./data/processed/idx2movie.pkl', 'rb') as f:
    idx2movie = pickle.load(f)

# Загружаем последовательности пользователей (только для теста, небольшую часть)
with open('./data/processed/user_sequences.pkl', 'rb') as f:
    user_sequences_full = pickle.load(f)

print(f"Всего пользователей в датасете: {len(user_sequences_full):,}")

# Ограничиваем для теста (1000 пользователей достаточно для оценки baseline)
TEST_USERS = 10000
user_ids = list(user_sequences_full.keys())[:TEST_USERS]
user_sequences = {uid: user_sequences_full[uid] for uid in user_ids}

print(f"Используем {len(user_sequences)} пользователей для оценки")

# Очищаем память от полной версии
del user_sequences_full
gc.collect()

# ============================================
# 2. СОЗДАНИЕ ТЕСТОВЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# ============================================
print("\n2. Создание тестовых последовательностей...")

test_sequences = []
for user_id, seq_data in user_sequences.items():
    pos_seq = seq_data['positive']
    if len(pos_seq) >= 3:
        test_sequences.append({
            'context': pos_seq[:-1],
            'target': pos_seq[-1]
        })

# Ограничиваем количество тестовых примеров
test_sequences = test_sequences[:20000]
print(f"Тестовых примеров: {len(test_sequences)}")

# ============================================
# 3. BASELINE 1: RANDOM (не требует предварительных данных)
# ============================================
print("\n" + "="*60)
print("BASELINE 1: RANDOM")
print("="*60)

class RandomBaseline:
    def __init__(self, all_movies):
        self.all_movies = list(all_movies)
        print(f"Всего фильмов: {len(self.all_movies)}")
    
    def recommend(self, context, top_k=10):
        # Исключаем фильмы из контекста
        candidates = [m for m in self.all_movies if m not in context]
        if len(candidates) < top_k:
            candidates = self.all_movies
        selected = random.sample(candidates, min(top_k, len(candidates)))
        return [(idx, random.random()) for idx in selected]

# Список всех фильмов
all_movies = list(movie2idx.values())
random_baseline = RandomBaseline(all_movies)

# ============================================
# 4. BASELINE 2: MOST POPULAR (требует статистики)
# ============================================
print("\n" + "="*60)
print("BASELINE 2: MOST POPULAR")
print("="*60)

class MostPopularBaseline:
    def __init__(self, user_sequences):
        print("Подсчет популярности фильмов...")
        movie_count = {}
        
        # Считаем частоту встречаемости фильмов
        for seq_data in tqdm(user_sequences.values(), desc="Подсчет"):
            for movie_idx in seq_data['positive']:
                movie_count[movie_idx] = movie_count.get(movie_idx, 0) + 1
        
        # Сортируем по популярности
        self.popular_movies = sorted(movie_count.keys(), key=lambda x: movie_count[x], reverse=True)[:100]
        print(f"Топ {len(self.popular_movies)} популярных фильмов")
        
        # Очищаем память
        del movie_count
        gc.collect()
    
    def recommend(self, context, top_k=10):
        return [(idx, 1.0) for idx in self.popular_movies[:top_k]]

popular_baseline = MostPopularBaseline(user_sequences)

# ============================================
# 5. BASELINE 3: ITEM-KNN (оптимизированная версия)
# ============================================
print("\n" + "="*60)
print("BASELINE 3: ITEM-KNN (ОПТИМИЗИРОВАННАЯ)")
print("="*60)

class OptimizedItemKNNBaseline:
    def __init__(self, ratings_file='./data/processed/ratings_processed.csv', 
                 movie2idx=None, idx2movie=None, k=50):
        
        print("Загрузка рейтингов чанками...")
        
        # Загружаем только необходимые колонки
        self.movie2idx = movie2idx
        self.idx2movie = idx2movie
        
        # Загружаем рейтинги чанками для экономии памяти
        chunk_size = 50000
        movie_popularity = {}
        
        # Сначала собираем популярные фильмы (те, у которых много оценок)
        for chunk in tqdm(pd.read_csv(ratings_file, chunksize=chunk_size), desc="Анализ рейтингов"):
            high_ratings = chunk[chunk['rating'] >= 4.0]
            for movie_id in high_ratings['movieId']:
                if movie_id in movie2idx:
                    movie_popularity[movie_id] = movie_popularity.get(movie_id, 0) + 1
        
        # Берем топ-500 самых популярных фильмов
        top_movies = sorted(movie_popularity.keys(), key=lambda x: movie_popularity[x], reverse=True)[:500]
        print(f"Выбрано {len(top_movies)} популярных фильмов для построения матрицы")
        
        # Загружаем матрицу пользователь-фильм только для топ фильмов
        print("Построение матрицы пользователь-фильм...")
        user_movie_data = {}
        
        for chunk in tqdm(pd.read_csv(ratings_file, chunksize=chunk_size), desc="Построение матрицы"):
            high_ratings = chunk[chunk['rating'] >= 4.0]
            
            for _, row in high_ratings.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                
                if movie_id in top_movies and movie_id in movie2idx:
                    if user_id not in user_movie_data:
                        user_movie_data[user_id] = set()
                    user_movie_data[user_id].add(movie_id)
        
        # Вычисляем похожесть фильмов (упрощенная версия)
        print("Вычисление похожести фильмов...")
        self.movie_similarity = {}
        movie_list = list(top_movies)
        
        for i, movie1 in enumerate(tqdm(movie_list, desc="Вычисление похожести")):
            users1 = user_movie_data.get(movie1, set())
            self.movie_similarity[movie1] = {}
            
            for j, movie2 in enumerate(movie_list[i+1:i+100]):  # Ограничиваем количество сравнений
                users2 = user_movie_data.get(movie2, set())
                if users1 and users2:
                    intersection = len(users1 & users2)
                    union = len(users1 | users2)
                    if union > 0:
                        similarity = intersection / union
                        if similarity > 0.1:  # Порог для экономии памяти
                            self.movie_similarity[movie1][movie2] = similarity
                            self.movie_similarity[movie2][movie1] = similarity
        
        print(f"Вычислено {sum(len(v) for v in self.movie_similarity.values())} связей")
        
        # Очищаем память
        del user_movie_data
        gc.collect()
    
    def recommend(self, context, top_k=10):
        if not context:
            return []
        
        # Берем последний фильм в контексте
        last_movie_idx = context[-1]
        last_movie_id = self.idx2movie.get(last_movie_idx)
        
        if last_movie_id not in self.movie_similarity:
            return []
        
        # Получаем похожие фильмы
        similar = self.movie_similarity[last_movie_id]
        sorted_similar = sorted(similar.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        recommendations = []
        for movie_id, score in sorted_similar:
            if movie_id in self.movie2idx:
                recommendations.append((self.movie2idx[movie_id], score))
        
        return recommendations

# Создаем ItemKNN baseline (может занять несколько минут)
try:
    knn_baseline = OptimizedItemKNNBaseline(
        ratings_file='./data/processed/ratings_processed.csv',
        movie2idx=movie2idx,
        idx2movie=idx2movie,
        k=50
    )
    knn_available = True
except Exception as e:
    print(f"ItemKNN не удалось создать: {e}")
    print("Используем упрощенную версию...")
    
    class SimpleItemKNN:
        def __init__(self, user_sequences):
            self.popular_movies = []
            movie_count = {}
            for seq_data in user_sequences.values():
                for movie_idx in seq_data['positive']:
                    movie_count[movie_idx] = movie_count.get(movie_idx, 0) + 1
            self.popular_movies = sorted(movie_count.keys(), key=lambda x: movie_count[x], reverse=True)[:100]
        
        def recommend(self, context, top_k=10):
            return [(idx, 1.0) for idx in self.popular_movies[:top_k]]
    
    knn_baseline = SimpleItemKNN(user_sequences)
    knn_available = False

# ============================================
# 6. ОЦЕНКА BASELINE МОДЕЛЕЙ
# ============================================
print("\n" + "="*60)
print("ОЦЕНКА BASELINE МОДЕЛЕЙ")
print("="*60)

def evaluate_baseline(baseline, test_sequences, name, k_values=[5, 10, 20]):
    print(f"\nОценка {name}...")
    results = {k: {'hits': 0, 'ndcg': 0} for k in k_values}
    total = 0
    
    for seq in tqdm(test_sequences, desc=name):
        context = seq['context']
        target = seq['target']
        
        recommendations = baseline.recommend(context, top_k=max(k_values))
        recommended_ids = [idx for idx, _ in recommendations]
        
        if target in recommended_ids:
            position = recommended_ids.index(target)
            
            for k in k_values:
                if position < k:
                    results[k]['hits'] += 1
                    results[k]['ndcg'] += 1.0 / np.log2(position + 2)
        
        total += 1
    
    for k in k_values:
        results[k]['hr'] = results[k]['hits'] / total
        results[k]['ndcg'] = results[k]['ndcg'] / total
    
    return results

# Оценка всех baseline
random_results = evaluate_baseline(random_baseline, test_sequences, "Random")
popular_results = evaluate_baseline(popular_baseline, test_sequences, "Most Popular")
knn_results = evaluate_baseline(knn_baseline, test_sequences, "Item-KNN")

# ============================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*60)

baseline_results = {
    'random': random_results,
    'most_popular': popular_results,
    'item_knn': knn_results
}

with open('./results/baseline_results.pkl', 'wb') as f:
    pickle.dump(baseline_results, f)

# ============================================
# 8. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ BASELINE МОДЕЛЕЙ")
print("="*60)

print("\n Random Baseline:")
for k in [5, 10, 20]:
    print(f"  HR@{k}: {random_results[k]['hr']:.4f}, NDCG@{k}: {random_results[k]['ndcg']:.4f}")

print("\n Most Popular Baseline:")
for k in [5, 10, 20]:
    print(f"  HR@{k}: {popular_results[k]['hr']:.4f}, NDCG@{k}: {popular_results[k]['ndcg']:.4f}")

print("\n Item-KNN Baseline:")
for k in [5, 10, 20]:
    print(f"  HR@{k}: {knn_results[k]['hr']:.4f}, NDCG@{k}: {knn_results[k]['ndcg']:.4f}")

# ============================================
# 9. ПРОСТАЯ ВИЗУАЛИЗАЦИЯ
# ============================================
print("\n9. Создание простой визуализации...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

k_values = [5, 10, 20]
models = ['Random', 'Most Popular', 'Item-KNN']
colors = ['gray', 'orange', 'blue']

# HR@K
for i, (model, color) in enumerate(zip(models, colors)):
    if model == 'Random':
        hr_values = [random_results[k]['hr'] for k in k_values]
    elif model == 'Most Popular':
        hr_values = [popular_results[k]['hr'] for k in k_values]
    else:
        hr_values = [knn_results[k]['hr'] for k in k_values]
    
    axes[0].plot(k_values, hr_values, marker='o', label=model, color=color, linewidth=2)

axes[0].set_xlabel('K')
axes[0].set_ylabel('Hit Rate (HR)')
axes[0].set_title('Сравнение Hit Rate@K')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# NDCG@K
for i, (model, color) in enumerate(zip(models, colors)):
    if model == 'Random':
        ndcg_values = [random_results[k]['ndcg'] for k in k_values]
    elif model == 'Most Popular':
        ndcg_values = [popular_results[k]['ndcg'] for k in k_values]
    else:
        ndcg_values = [knn_results[k]['ndcg'] for k in k_values]
    
    axes[1].plot(k_values, ndcg_values, marker='s', label=model, color=color, linewidth=2)

axes[1].set_xlabel('K')
axes[1].set_ylabel('NDCG')
axes[1].set_title('Сравнение NDCG@K')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/baseline_comparison.png', dpi=150)
plt.show()

print("\n Baseline результаты сохранены:")
print("  - ./results/baseline_results.pkl")
print("  - ./results/baseline_comparison.png")

print("\n" + "="*60)
print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО!")
print("="*60)