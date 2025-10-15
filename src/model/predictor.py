"""
Предсказание результатов с учётом спринтов (улучшенная версия)
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.loader import F1DataLoader
from src.visualization.plotter import ResultsPlotter
from config import POINTS_SYSTEM, F1_CALENDAR

class F1Predictor:
    def __init__(self):
        """Инициализация предиктора"""
        self.loader = F1DataLoader()
        self.plotter = ResultsPlotter()

    def test_model(self, model_path, test_year):
        """Тестирование модели на данных за конкретный год"""
        print(f"\n📥 Загрузка модели...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        cleaner = model_data['cleaner']

        print(f"\n📊 Загрузка данных за {test_year}...")
        test_data = self.loader.load_season_data(test_year)

        print(f"\n🔮 Генерация предсказаний...")
        x_test, processed_df = cleaner.prepare_prediction_data(test_data)
        predictions = model.predict(x_test)

        # Округляем и ограничиваем предсказания
        predictions = np.clip(np.round(predictions).astype(int), 1, 20)

        processed_df['predicted_position'] = predictions

        # Вычисляем метрики
        actual = processed_df['position'].values
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        top10_accuracy = np.mean((actual <= 10) == (predictions <= 10))

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'accuracy': top10_accuracy
        }

        # Агрегация результатов
        from src.data.cleaner import DataCleaner
        temp_cleaner = DataCleaner()

        # Реальные результаты
        actual_drivers, actual_teams = temp_cleaner.aggregate_season_results(processed_df)

        # Предсказанные результаты
        pred_df = processed_df.copy()
        pred_df['position'] = pred_df['predicted_position']
        predicted_drivers, predicted_teams = temp_cleaner.aggregate_season_results(pred_df)

        # Создаем директорию для результатов
        output_dir = Path("data") / f"test_{test_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем результаты
        self._save_test_results(
            output_dir, processed_df,
            actual_drivers, actual_teams,
            predicted_drivers, predicted_teams,
            metrics, test_year
        )

        # Визуализация
        print(f"\n📊 Создание визуализаций...")
        self.plotter.plot_test_results(
            actual_drivers, predicted_drivers,
            actual_teams, predicted_teams,
            output_dir
        )

        return {
            'metrics': metrics,
            'output_dir': str(output_dir)
        }

    def predict_season(self, model_path, predict_year):
        """
        УЛУЧШЕННОЕ предсказание результатов на будущий сезон
        - Предсказание для каждой гонки отдельно
        - Учет специфики трасс
        - Учет спринтов с правильным начислением очков
        - Добавление вариативности
        """
        print(f"\n📥 Загрузка модели...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        cleaner = model_data['cleaner']
        train_years = model_data['metrics']['years']

        # Получаем данные за последний доступный год
        last_available_year = predict_year - 1
        print(f"\n📊 Использование данных за {last_available_year} как базу...")

        try:
            base_data = self.loader.load_season_data(last_available_year)
        except Exception as e:
            last_available_year = max(train_years)
            print(f"⚠️  Используются данные за {last_available_year}: {e}")
            base_data = self.loader.load_season_data(last_available_year)

        # Получаем активных гонщиков и команды
        active_drivers, active_teams = self.loader.get_active_drivers_teams(last_available_year)

        print(f"\n👥 Активных гонщиков: {len(active_drivers)}")
        print(f"🏢 Активных команд: {len(active_teams)}")
        print(f"🏎️  Гонок в сезоне: {len(F1_CALENDAR)}")

        if len(active_drivers) == 0:
            raise ValueError(f"Не найдено активных гонщиков за {last_available_year} год!")

        # Фильтруем данные
        base_data = base_data[base_data['driver'].isin(active_drivers)].copy()

        # Создаем базовые характеристики гонщиков
        driver_profiles = self._create_driver_profiles(base_data, active_drivers)

        # Предсказываем каждую гонку И спринт отдельно!
        print(f"\n🏁 Симуляция {len(F1_CALENDAR)} этапов (с учётом спринтов)...")
        season_results = self._simulate_season(
            model, cleaner, driver_profiles,
            F1_CALENDAR, predict_year
        )

        # Агрегируем результаты по гонщикам
        driver_season_stats = self._aggregate_driver_results(season_results)

        # Агрегируем по командам
        team_season_stats = self._aggregate_team_results(driver_season_stats)

        # Топ-10 гонщиков и топ-3 команды
        top_drivers = driver_season_stats.head(10).to_dict('records')
        top_teams = team_season_stats.head(3).to_dict('records')

        # Создаем директорию для результатов
        years_str = '-'.join(map(str, train_years))
        output_dir = Path("data") / f"prediction_{predict_year}_trained_{years_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем результаты
        self._save_prediction_results_enhanced(
            output_dir, driver_season_stats, team_season_stats,
            season_results, top_drivers, top_teams,
            predict_year, train_years
        )

        # Визуализация
        print(f"\n📊 Создание визуализаций...")
        self.plotter.plot_predictions(
            driver_season_stats, team_season_stats,
            output_dir, predict_year
        )

        return {
            'top_drivers': top_drivers,
            'top_teams': top_teams,
            'season_results': season_results,
            'output_dir': str(output_dir)
        }

    @staticmethod
    def _create_driver_profiles(base_data, active_drivers):
        """Создание профилей гонщиков с их характеристиками"""
        profiles = {}

        for driver in active_drivers:
            driver_data = base_data[base_data['driver'] == driver]

            if len(driver_data) == 0:
                continue

            # Безопасное извлечение значений
            def safe_get(df, col, default):
                if col in df.columns and len(df) > 0:
                    value = df.iloc[0][col]
                    return value if pd.notna(value) else default
                return default

            def safe_mean(df, col, default):
                if col in df.columns:
                    mean_val = df[col].mean()
                    return mean_val if pd.notna(mean_val) else default
                return default

            def safe_std(df, col, default=1.0):
                if col in df.columns and len(df) > 1:
                    std_val = df[col].std()
                    return std_val if pd.notna(std_val) and std_val > 0 else default
                return default

            profiles[driver] = {
                # Базовая информация
                'driver': driver,
                'team': safe_get(driver_data, 'team', 'Unknown'),
                'driver_abbr': safe_get(driver_data, 'driver_abbr', 'UNK'),

                # Средние показатели
                'avg_quali_position': safe_mean(driver_data, 'quali_position', 10.0),
                'avg_grid_position': safe_mean(driver_data, 'grid_position', 10.0),
                'avg_position': safe_mean(driver_data, 'position', 10.0),
                'avg_q1_time': safe_mean(driver_data, 'q1_time', 75.0),
                'avg_q2_time': safe_mean(driver_data, 'q2_time', 74.0),
                'avg_q3_time': safe_mean(driver_data, 'q3_time', 73.0),
                'avg_fastest_lap': safe_mean(driver_data, 'fastest_lap', 75.0),
                'avg_lap_time': safe_mean(driver_data, 'avg_lap_time', 76.0),
                'avg_pit_stops': safe_mean(driver_data, 'pit_stops', 2.0),
                'avg_speed': safe_mean(driver_data, 'avg_speed', 150.0),

                # Вариативность (для добавления шума)
                'quali_std': safe_std(driver_data, 'quali_position', 2.0),
                'position_std': safe_std(driver_data, 'position', 3.0),

                # Статистика
                'total_races': len(driver_data),
                'dnf_rate': safe_mean(driver_data, 'dnf', 0.0),
            }

        return profiles

    def _simulate_season(self, model, cleaner, driver_profiles, calendar, year):
        """Симуляция всего сезона: предсказание каждой гонки и спринта"""
        season_results = []

        for race_idx, race in enumerate(calendar, 1):
            race_name = race['name']
            has_sprint = race.get('has_sprint', False)
            sprint_predictions = []

            sprint_str = " + Sprint" if has_sprint else ""
            print(f"  🏁 [{race_idx}/{len(calendar)}] {race_name}{sprint_str}... ", end='')

            # Если есть спринт, сначала симулируем его
            if has_sprint:
                sprint_predictions = self._simulate_race_session(
                    model, cleaner, driver_profiles, race, year, is_sprint=True
                )
                season_results.extend(sprint_predictions)

            # Симулируем основную гонку
            race_predictions = self._simulate_race_session(
                model, cleaner, driver_profiles, race, year, is_sprint=False
            )
            season_results.extend(race_predictions)

            total_predictions = len(sprint_predictions) if has_sprint else 0
            total_predictions += len(race_predictions)
            print(f"✓ ({total_predictions} результатов)")

        return pd.DataFrame(season_results)

    def _simulate_race_session(self, model, cleaner, driver_profiles, race, year, is_sprint=False):
        """Симуляция одной сессии (гонка или спринт)"""
        session_predictions = []
        session_type = 'sprint' if is_sprint else 'race'

        for driver, profile in driver_profiles.items():
            # Создаем данные для этой конкретной сессии
            race_data = self._create_race_data(profile, race, year, is_sprint)

            # Подготавливаем данные
            race_df = pd.DataFrame([race_data])

            try:
                (x, _) = cleaner.prepare_prediction_data(race_df)

                # Базовое предсказание модели
                base_prediction = model.predict(x)[0]

                # УЛУЧШЕНИЕ 1: Корректировка по типу трассы
                track_adjusted = self._adjust_for_track_type(
                    base_prediction, profile, race
                )

                # УЛУЧШЕНИЕ 2: Добавление реалистичной вариативности
                final_position = self._add_race_variability(
                    track_adjusted, profile, is_sprint
                )

                # Ограничиваем позицию
                final_position = int(np.clip(np.round(final_position), 1, 20))

                # Расчет очков по правильной системе
                points = self._get_points_for_position(final_position, is_sprint)

                session_predictions.append({
                    'driver': driver,
                    'team': profile['team'],
                    'race': race['name'],
                    'session_type': session_type,
                    'predicted_position': final_position,
                    'points': points,
                    'base_prediction': base_prediction,
                    'track_type': race['track_type']
                })

            except Exception as e:
                print(f"\n      ⚠️  Ошибка для {driver} ({session_type}): {e}")
                continue

        return session_predictions

    @staticmethod
    def _create_race_data(profile, race, year, is_sprint=False):
        """Создание данных для конкретной сессии"""
        return {
            'year': year,
            'event': race['name'],
            'session_type': 'sprint' if is_sprint else 'race',
            'driver': profile['driver'],
            'driver_abbr': profile['driver_abbr'],
            'team': profile['team'],
            'grid_position': profile['avg_grid_position'],
            'quali_position': profile['avg_quali_position'],
            'q1_time': profile['avg_q1_time'],
            'q2_time': profile['avg_q2_time'],
            'q3_time': profile['avg_q3_time'],
            'fastest_lap': profile['avg_fastest_lap'],
            'avg_lap_time': profile['avg_lap_time'],
            'pit_stops': profile['avg_pit_stops'],
            'avg_speed': profile['avg_speed'],
            'air_temp': 25.0,
            'track_temp': 35.0,
            'humidity': 50.0,
            'rainfall': 0,
            'dnf': 0,
            'dsq': 0,
            'is_sprint': 1 if is_sprint else 0,
            'position': np.nan,
            'points': 0,
            'status': 'Prediction'
        }

    @staticmethod
    def _adjust_for_track_type(base_prediction, profile, race):
        """Корректировка предсказания по характеристикам трассы"""
        quali_importance = race['quali_importance']

        # Если квалификация очень важна (Monaco, Singapore)
        if quali_importance > 0.85:
            # Гонщики с хорошей квалификацией получают преимущество
            quali_factor = (10 - profile['avg_quali_position']) / 10
            adjustment = quali_factor * 1.5
            return base_prediction - adjustment

        # Если квалификация менее важна (Monza, Spa)
        elif quali_importance < 0.65:
            # Больше шансов для обгонов, меньше зависимость от квалификации
            adjustment = np.random.normal(0, 1.5)
            return base_prediction + adjustment

        # Стандартные трассы
        return base_prediction

    @staticmethod
    def _add_race_variability(position, profile, is_sprint=False):
        """Добавление реалистичной вариативности"""
        # Базовый шум (каждая гонка уникальна)
        # Спринты более предсказуемы (меньше вариативности)
        noise_multiplier = 0.5 if is_sprint else 0.7
        race_noise = np.random.normal(0, profile['position_std'] * noise_multiplier)

        # Случайные события
        random_factor = np.random.random()

        # 3% шанс DNF/большая проблема (меньше в спринтах)
        dnf_chance = profile['dnf_rate'] * 1.2 if not is_sprint else profile['dnf_rate'] * 0.5
        if random_factor < dnf_chance:
            return np.random.uniform(15, 20)

        # 2% шанс выдающейся гонки (на 2-3 позиции лучше)
        elif random_factor > 0.98:
            brilliance = np.random.uniform(-3, -2)
            return position + brilliance

        # 5% шанс плохой гонки (на 2-4 позиции хуже)
        elif random_factor < 0.05:
            bad_race = np.random.uniform(2, 4)
            return position + bad_race

        # Обычная гонка с естественной вариативностью
        return position + race_noise

    @staticmethod
    def _get_points_for_position(position, is_sprint=False):
        """НОВОЕ: Расчет очков за позицию по правильной системе F1"""
        if is_sprint:
            return POINTS_SYSTEM['sprint'].get(position, 0)
        else:
            return POINTS_SYSTEM['race'].get(position, 0)

    @staticmethod
    def _aggregate_driver_results(season_results):
        """Агрегация результатов по гонщикам"""
        driver_stats = season_results.groupby('driver').agg({
            'points': 'sum',
            'predicted_position': 'mean',
            'team': 'first'
        }).reset_index()

        driver_stats.columns = ['driver', 'total_points', 'avg_position', 'team']

        # Дополнительная статистика
        for driver in driver_stats['driver']:
            driver_races = season_results[season_results['driver'] == driver]

            # Считаем только основные гонки для побед (не спринты)
            main_races = driver_races[driver_races['session_type'] == 'race']

            wins = len(main_races[main_races['predicted_position'] == 1])
            podiums = len(main_races[main_races['predicted_position'] <= 3])
            points_finishes = len(main_races[main_races['predicted_position'] <= 10])

            # Спринты отдельно
            sprint_races = driver_races[driver_races['session_type'] == 'sprint']
            sprint_wins = len(sprint_races[sprint_races['predicted_position'] == 1])

            driver_stats.loc[driver_stats['driver'] == driver, 'wins'] = wins
            driver_stats.loc[driver_stats['driver'] == driver, 'podiums'] = podiums
            driver_stats.loc[driver_stats['driver'] == driver, 'points_finishes'] = points_finishes
            driver_stats.loc[driver_stats['driver'] == driver, 'sprint_wins'] = sprint_wins

        driver_stats = driver_stats.sort_values('total_points', ascending=False).reset_index(drop=True)
        return driver_stats

    @staticmethod
    def _aggregate_team_results(driver_stats):
        """Агрегация результатов по командам"""
        team_stats = driver_stats.groupby('team').agg({
            'total_points': 'sum',
            'avg_position': 'mean',
            'wins': 'sum',
            'podiums': 'sum'
        }).reset_index()

        team_stats.columns = ['team', 'total_points', 'avg_position', 'wins', 'podiums']
        team_stats = team_stats.sort_values('total_points', ascending=False).reset_index(drop=True)
        return team_stats

    @staticmethod
    def _save_test_results(output_dir, full_data, actual_drivers, actual_teams,
                           pred_drivers, pred_teams, metrics, year):
        """Сохранение результатов тестирования"""
        # CSV с полными данными
        full_data.to_csv(output_dir / "full_test_results.csv", index=False, encoding='utf-8')

        # CSV с реальными результатами
        actual_drivers.to_csv(output_dir / "actual_drivers.csv", index=False, encoding='utf-8')
        actual_teams.to_csv(output_dir / "actual_teams.csv", index=False, encoding='utf-8')

        # CSV с предсказаниями
        pred_drivers.to_csv(output_dir / "predicted_drivers.csv", index=False, encoding='utf-8')
        pred_teams.to_csv(output_dir / "predicted_teams.csv", index=False, encoding='utf-8')

        # JSON с метриками и топами
        results_json = {
            'year': year,
            'metrics': metrics,
            'top10_actual': actual_drivers.head(10).to_dict('records'),
            'top10_predicted': pred_drivers.head(10).to_dict('records'),
            'top3_teams_actual': actual_teams.head(3).to_dict('records'),
            'top3_teams_predicted': pred_teams.head(3).to_dict('records'),
        }

        with open(output_dir / "test_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        print(f"✅ Результаты сохранены в: {output_dir}")

    @staticmethod
    def _save_prediction_results_enhanced(output_dir, driver_stats, team_stats,
                                          season_results, top_drivers, top_teams,
                                          predict_year, train_years):
        """Сохранение улучшенных результатов предсказания"""
        # CSV с результатами гонщиков
        driver_stats.to_csv(output_dir / "predicted_drivers.csv", index=False, encoding='utf-8')
        team_stats.to_csv(output_dir / "predicted_teams.csv", index=False, encoding='utf-8')

        # CSV со всеми гонками (включая спринты)
        season_results.to_csv(output_dir / "season_races_detailed.csv", index=False, encoding='utf-8')

        # Подсчитываем статистику спринтов
        sprint_count = len(season_results[season_results['session_type'] == 'sprint']) // 20  # примерное количество гонщиков
        race_count = len(F1_CALENDAR)

        # JSON с полными результатами
        prediction_json = {
            'prediction_year': predict_year,
            'trained_on_years': train_years,
            'generated_at': datetime.now().isoformat(),
            'prediction_method': 'enhanced_per_race_with_sprints_and_variability',
            'total_races': race_count,
            'total_sprints': sprint_count,
            'points_system': POINTS_SYSTEM,
            'top10_drivers': top_drivers,
            'top3_teams': top_teams,
            'all_drivers': driver_stats.to_dict('records'),
            'all_teams': team_stats.to_dict('records'),
            'race_calendar': F1_CALENDAR,
        }

        with open(output_dir / "predictions.json", 'w', encoding='utf-8') as f:
            json.dump(prediction_json, f, indent=2, ensure_ascii=False)

        # JSON с детализацией по гонкам для каждого гонщика
        drivers_race_details = {}
        for driver in season_results['driver'].unique():
            driver_races = season_results[season_results['driver'] == driver]
            drivers_race_details[driver] = driver_races[
                ['race', 'session_type', 'predicted_position', 'points']
            ].to_dict('records')

        with open(output_dir / "drivers_race_by_race.json", 'w', encoding='utf-8') as f:
            json.dump(drivers_race_details, f, indent=2, ensure_ascii=False)

        print(f"✅ Предсказания сохранены в: {output_dir}")
        print(f"   📄 predictions.json - основные результаты")
        print(f"   📄 season_races_detailed.csv - детали каждой гонки и спринта")
        print(f"   📄 drivers_race_by_race.json - гонка за гонкой для каждого гонщика")
        print(f"   📊 Всего гонок: {race_count}, спринтов: {sprint_count}")