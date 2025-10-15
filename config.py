"""
Конфигурационные параметры проекта
"""

# Настройки кэша FastF1
CACHE_DIR = "./f1_cache"

# Защита от rate limiting API (задержка между запросами в секундах)
# Если получаете ошибки 429, увеличьте это значение
API_REQUEST_DELAY = 2.0

# Сезоны для анализа и тренировки по умолчанию
DEFAULT_TRAIN_YEARS = [2019, 2021, 2022, 2023, 2024]
PREDICT_YEAR = 2025

# Настройки модели
MODEL_CONFIG = {
    'test_size': 0.15,
    'random_state': 42,
    'n_estimators': 400,
    'max_depth': 15,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'validation_fraction': 0.05,
    'n_iter_no_change': 15,
    'verbose': 1,
}

# Очки за позиции (F1 2024+ правила)
POINTS_SYSTEM = {
    'race': {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    },
    'sprint': {
        1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
        6: 3, 7: 2, 8: 1
    }
}

# Календарь F1 2025 с информацией о спринтах
F1_CALENDAR = [
    {'name': 'Bahrain Grand Prix', 'country': 'Bahrain', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 4, 'quali_importance': 0.65},
    {'name': 'Saudi Arabian Grand Prix', 'country': 'Saudi Arabia', 'has_sprint': False, 'track_type': 'street', 'overtaking_difficulty': 7, 'quali_importance': 0.80},
    {'name': 'Australian Grand Prix', 'country': 'Australia', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Japanese Grand Prix', 'country': 'Japan', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 6, 'quali_importance': 0.75},
    {'name': 'Chinese Grand Prix', 'country': 'China', 'has_sprint': True, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Miami Grand Prix', 'country': 'USA', 'has_sprint': True, 'track_type': 'street', 'overtaking_difficulty': 6, 'quali_importance': 0.75},
    {'name': 'Emilia Romagna Grand Prix', 'country': 'Italy', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 7, 'quali_importance': 0.80},
    {'name': 'Monaco Grand Prix', 'country': 'Monaco', 'has_sprint': False, 'track_type': 'street', 'overtaking_difficulty': 10, 'quali_importance': 0.95},
    {'name': 'Spanish Grand Prix', 'country': 'Spain', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 6, 'quali_importance': 0.70},
    {'name': 'Canadian Grand Prix', 'country': 'Canada', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Austrian Grand Prix', 'country': 'Austria', 'has_sprint': True, 'track_type': 'circuit', 'overtaking_difficulty': 4, 'quali_importance': 0.65},
    {'name': 'British Grand Prix', 'country': 'Great Britain', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Belgian Grand Prix', 'country': 'Belgium', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 3, 'quali_importance': 0.60},
    {'name': 'Hungarian Grand Prix', 'country': 'Hungary', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 8, 'quali_importance': 0.85},
    {'name': 'Dutch Grand Prix', 'country': 'Netherlands', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 7, 'quali_importance': 0.75},
    {'name': 'Italian Grand Prix', 'country': 'Italy', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 3, 'quali_importance': 0.60},
    {'name': 'Azerbaijan Grand Prix', 'country': 'Azerbaijan', 'has_sprint': False, 'track_type': 'street', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Singapore Grand Prix', 'country': 'Singapore', 'has_sprint': False, 'track_type': 'street', 'overtaking_difficulty': 8, 'quali_importance': 0.85},
    {'name': 'United States Grand Prix', 'country': 'USA', 'has_sprint': True, 'track_type': 'circuit', 'overtaking_difficulty': 4, 'quali_importance': 0.65},
    {'name': 'Mexico City Grand Prix', 'country': 'Mexico', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'São Paulo Grand Prix', 'country': 'Brazil', 'has_sprint': True, 'track_type': 'circuit', 'overtaking_difficulty': 4, 'quali_importance': 0.65},
    {'name': 'Las Vegas Grand Prix', 'country': 'USA', 'has_sprint': False, 'track_type': 'street', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Qatar Grand Prix', 'country': 'Qatar', 'has_sprint': True, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
    {'name': 'Abu Dhabi Grand Prix', 'country': 'UAE', 'has_sprint': False, 'track_type': 'circuit', 'overtaking_difficulty': 5, 'quali_importance': 0.70},
]