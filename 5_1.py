# 6.py - Полный пайплайн рекомендательной системы на Transformer Decoder
# Запускать одним файлом!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gc
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================
print("="*60)
print("1. ЗАГРУЗКА КОНФИГУРАЦИИ")
print("="*60)

with open('./data/processed/config.pkl', 'rb') as f:
    config_data = pickle.load(f)

PAD_TOKEN = config_data['pad_token']
BOS_TOKEN = config_data['bos_token']
EOS_TOKEN = config_data['eos_token']
UNK_TOKEN = config_data['unk_token']
NUM_SPECIAL_TOKENS = config_data['num_special_tokens']
num_movies = config_data['num_movies']

print(f"Размер словаря: {num_movies}")
print(f"PAD_TOKEN: {PAD_TOKEN}, BOS_TOKEN: {BOS_TOKEN}, EOS_TOKEN: {EOS_TOKEN}")

# ============================================
# 2. ЗАГРУЗКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ ПОЛЬЗОВАТЕЛЕЙ
# ============================================
print("\n" + "="*60)
print("2. ЗАГРУЗКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ ПОЛЬЗОВАТЕЛЕЙ")
print("="*60)

with open('./data/processed/user_sequences.pkl', 'rb') as f:
    user_sequences = pickle.load(f)

print(f"Всего пользователей: {len(user_sequences):,}")

# Ограничиваем количество пользователей для экономии памяти
MAX_USERS = 30000
if len(user_sequences) > MAX_USERS:
    user_ids = list(user_sequences.keys())[:MAX_USERS]
    user_sequences = {uid: user_sequences[uid] for uid in user_ids}
    print(f"Ограничено до {MAX_USERS} пользователей")

# ============================================
# 3. LAZY DATASET (создание примеров на лету)
# ============================================
print("\n" + "="*60)
print("3. СОЗДАНИЕ LAZY DATASET")
print("="*60)

class LazyRecommenderDataset(Dataset):
    """Датасет с ленивой загрузкой - не хранит все примеры в памяти"""
    
    def __init__(self, user_sequences, seq_len=20, pad_token=0, bos_token=1, 
                 max_examples=None, is_train=True):
        self.seq_len = seq_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.is_train = is_train
        self.index_map = []
        
        # Получаем список всех фильмов для отрицательных примеров
        all_movies = set()
        for seq_data in user_sequences.values():
            all_movies.update(seq_data['positive'])
        self.all_movies = list(all_movies)
        
        print("Индексация примеров...")
        for user_id, seq_data in tqdm(user_sequences.items(), desc="Индексация"):
            pos_seq = seq_data['positive']
            if len(pos_seq) < 3:
                continue
            
            if is_train:
                # Для обучения: все кроме последнего фильма
                indices = range(1, len(pos_seq))
            else:
                # Для валидации: только последний фильм
                indices = [len(pos_seq) - 1]
            
            for i in indices:
                context = pos_seq[max(0, i - seq_len):i]
                target = pos_seq[i]
                self.index_map.append({
                    'context': context,
                    'target': target,
                    'neg_samples': seq_data.get('negative', [])
                })
                
                if max_examples and len(self.index_map) >= max_examples:
                    break
            
            if max_examples and len(self.index_map) >= max_examples:
                break
        
        print(f"Создано {len(self.index_map):,} индексов примеров")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        item = self.index_map[idx]
        context = item['context']
        target = item['target']
        
        # Для обучения иногда заменяем target на отрицательный пример
        if self.is_train and item['neg_samples'] and random.random() < 0.3:
            neg_samples = item['neg_samples']
            if neg_samples:
                target = random.choice(neg_samples)
        
        # Формируем входную последовательность
        input_seq = [self.bos_token] + context
        if len(input_seq) < self.seq_len + 1:
            input_seq = input_seq + [self.pad_token] * (self.seq_len + 1 - len(input_seq))
        else:
            input_seq = input_seq[:self.seq_len + 1]
        
        # Labels: -100 для всех позиций, кроме последней
        labels = [-100] * len(input_seq)
        labels[-1] = target
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

# Параметры датасета
SEQ_LEN = 20
TRAIN_RATIO = 0.8

# Разделяем пользователей на train и val
user_ids = list(user_sequences.keys())
random.seed(42)
random.shuffle(user_ids)

split_idx = int(len(user_ids) * TRAIN_RATIO)
train_user_ids = user_ids[:split_idx]
val_user_ids = user_ids[split_idx:]

train_sequences = {uid: user_sequences[uid] for uid in train_user_ids}
val_sequences = {uid: user_sequences[uid] for uid in val_user_ids}

print(f"Train пользователей: {len(train_sequences):,}")
print(f"Val пользователей: {len(val_sequences):,}")

# Создаем Lazy датасеты
train_dataset = LazyRecommenderDataset(
    train_sequences, 
    seq_len=SEQ_LEN, 
    pad_token=PAD_TOKEN, 
    bos_token=BOS_TOKEN,
    max_examples=300000,  # Ограничиваем для экономии памяти
    is_train=True
)

val_dataset = LazyRecommenderDataset(
    val_sequences, 
    seq_len=SEQ_LEN, 
    pad_token=PAD_TOKEN, 
    bos_token=BOS_TOKEN,
    max_examples=50000,
    is_train=False
)

print(f"\nTrain samples: {len(train_dataset):,}")
print(f"Val samples: {len(val_dataset):,}")

# ============================================
# 4. СОЗДАНИЕ DATALOADER'ов
# ============================================
print("\n" + "="*60)
print("4. СОЗДАНИЕ DATALOADER'ов")
print("="*60)

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

BATCH_SIZE = 32
NUM_WORKERS = 0  # Для Windows

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE * 2, 
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

print(f"Train batches: {len(train_loader):,}")
print(f"Val batches: {len(val_loader):,}")

# ============================================
# 5. АРХИТЕКТУРА МОДЕЛИ
# ============================================
print("\n" + "="*60)
print("5. СОЗДАНИЕ МОДЕЛИ")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

class RecommenderGPT(nn.Module):
    def __init__(self, num_movies, embed_dim=256, num_heads=8, num_layers=6, 
                 max_seq_len=50, dropout=0.1, pad_token_id=0):
        super().__init__()
        
        config = GPT2Config(
            vocab_size=num_movies,
            n_embd=embed_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_len,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=BOS_TOKEN,
            eos_token_id=EOS_TOKEN,
            pad_token_id=pad_token_id
        )
        
        self.transformer = GPT2LMHeadModel(config)
        self.config = {
            'num_movies': num_movies,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'max_seq_len': max_seq_len,
            'dropout': dropout
        }
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Всего параметров: {total_params:,}")
    
    def forward(self, input_ids, labels=None):
        return self.transformer(input_ids=input_ids, labels=labels)

# Создание модели
model = RecommenderGPT(
    num_movies=num_movies,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=SEQ_LEN + 1,
    pad_token_id=PAD_TOKEN
).to(device)

# ============================================
# 6. НАСТРОЙКА ОБУЧЕНИЯ
# ============================================
print("\n" + "="*60)
print("6. НАСТРОЙКА ОБУЧЕНИЯ")
print("="*60)

EPOCHS = 10
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=WARMUP_STEPS, 
    num_training_steps=total_steps
)

print(f"Эпох: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Всего шагов: {total_steps:,}")

# ============================================
# 7. ОБУЧЕНИЕ
# ============================================
print("\n" + "="*60)
print("7. НАЧАЛО ОБУЧЕНИЯ")
print("="*60)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Обучение
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Валидация
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    print("-" * 50)
    
    # Очистка памяти
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nОбучение завершено!")

# ============================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*60)

# Сохранение модели
os.makedirs('./models', exist_ok=True)
torch.save(model.state_dict(), './models/final_recommender_model.pt')
print("Модель сохранена в './models/final_recommender_model.pt'")

# Сохранение истории обучения
history = {
    'train_losses': train_losses,
    'val_losses': val_losses
}
with open('./results/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print("История обучения сохранена")

# ============================================
# 9. ВИЗУАЛИЗАЦИЯ
# ============================================
print("\n" + "="*60)
print("9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*60)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Кривые обучения')
plt.legend()
plt.grid(True, alpha=0.3)
os.makedirs('./results', exist_ok=True)
plt.savefig('./results/training_curves.png', dpi=150)
plt.show()

print("\nФинальные результаты:")
print(f"  Лучшая Val Loss: {min(val_losses):.4f}")
print(f"  Финальная Train Loss: {train_losses[-1]:.4f}")
print(f"  Финальная Val Loss: {val_losses[-1]:.4f}")

print("\n" + "="*60)
print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("="*60)
