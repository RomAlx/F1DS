"""
Загрузка данных из FastF1 с поддержкой спринтов и защитой от rate limiting
"""
import fastf1
import pandas as pd
import numpy as np
import time
from pathlib import Path
from config import CACHE_DIR, API_REQUEST_DELAY

class F1DataLoader:
    def __init__(self):
        """Инициализация загрузчика данных"""
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

        # Маппинг команд для учета переименований
        self.team_mappings = {
            'Racing Point': 'Aston Martin',
            'Alfa Romeo': 'Sauber',
            'AlphaTauri': 'RB',
            'Renault': 'Alpine',
        }

        # Задержка между запросами (из конфига)
        self.request_delay = API_REQUEST_DELAY

    def normalize_team_name(self, team_name):
        """Нормализация названия команды с учетом переименований"""
        return self.team_mappings.get(team_name, team_name)

    def cache_season(self, year):
        """Кэширование данных за сезон (включая спринты)"""
        print(f"\n📥 Загрузка календаря сезона {year}...")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"❌ Ошибка загрузки календаря: {e}")
            return

        total_events = len(schedule)
        print(f"📊 Всего этапов в сезоне: {total_events}")
        print(f"⏱️  Задержка между запросами: {self.request_delay}s (защита от rate limiting)")

        for idx, event in schedule.iterrows():
            event_name = event['EventName']

            try:
                print(f"\n  [{idx+1}/{total_events}] 🏁 {event_name}")

                # Загружаем квалификацию
                print(f"    ⏳ Загрузка квалификации...", end='', flush=True)
                quali = fastf1.get_session(year, event_name, 'Q')
                quali.load()
                print(" ✓")
                time.sleep(self.request_delay)

                # Проверяем и загружаем спринт если есть
                try:
                    print(f"    ⏳ Проверка спринта...", end='', flush=True)
                    sprint = fastf1.get_session(year, event_name, 'S')
                    sprint.load()
                    print(" ✓ Спринт найден")
                    time.sleep(self.request_delay)

                    # Загружаем Sprint Qualifying если есть (2023+)
                    try:
                        print(f"    ⏳ Загрузка Sprint Qualifying...", end='', flush=True)
                        sprint_quali = fastf1.get_session(year, event_name, 'SQ')
                        sprint_quali.load()
                        print(" ✓")
                        time.sleep(self.request_delay)
                    except Exception:
                        print(" - (нет SQ)")

                except Exception:
                    print(" - (нет спринта)")

                # Загружаем гонку
                print(f"    ⏳ Загрузка гонки...", end='', flush=True)
                session = fastf1.get_session(year, event_name, 'R')
                session.load()
                print(" ✓")
                time.sleep(self.request_delay)

            except Exception as e:
                print(f"\n    ⚠️  Ошибка при загрузке {event_name}: {e}")
                print(f"    💡 Пропускаем и продолжаем...")
                time.sleep(self.request_delay * 2)
                continue

        print(f"\n✅ Сезон {year} успешно закэширован!")

    def load_season_data(self, year):
        """Загрузка всех данных за сезон (включая спринты)"""
        print(f"\n📊 Сбор данных за {year} год...")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"❌ Ошибка загрузки календаря: {e}")
            return pd.DataFrame()

        season_data = []
        successful_events = 0
        failed_events = 0

        for idx, event in schedule.iterrows():
            event_name = event['EventName']

            try:
                # Загружаем квалификацию
                quali = fastf1.get_session(year, event_name, 'Q')
                quali.load()
                time.sleep(self.request_delay)

                # Загружаем гонку
                session = fastf1.get_session(year, event_name, 'R')
                session.load()
                time.sleep(self.request_delay)

                # Собираем данные по гонке
                race_data = self._extract_race_data(session, quali, event_name, year, is_sprint=False)
                season_data.extend(race_data)

                # Проверяем наличие спринта
                try:
                    sprint = fastf1.get_session(year, event_name, 'S')
                    sprint.load()
                    time.sleep(self.request_delay)

                    # Пытаемся загрузить Sprint Qualifying
                    try:
                        sprint_quali = fastf1.get_session(year, event_name, 'SQ')
                        sprint_quali.load()
                        time.sleep(self.request_delay)
                    except:
                        # Для старых сезонов используем обычную квалификацию
                        sprint_quali = quali

                    sprint_data = self._extract_race_data(sprint, sprint_quali, event_name, year, is_sprint=True)
                    season_data.extend(sprint_data)
                    print(f"  ✓ {event_name} (+ спринт)")

                except Exception:
                    # Нет спринта - это нормально
                    print(f"  ✓ {event_name}")

                successful_events += 1

            except Exception as e:
                print(f"  ⚠️  Пропуск {event_name}: {e}")
                failed_events += 1
                time.sleep(self.request_delay * 2)
                continue

        df = pd.DataFrame(season_data)
        print(f"\n✅ Собрано {len(df)} записей")
        print(f"   📊 Успешно загружено: {successful_events} этапов")
        if failed_events > 0:
            print(f"   ⚠️  Пропущено: {failed_events} этапов")

        return df

    def _extract_race_data(self, race_session, quali_session, event_name, year, is_sprint=False):
        """Извлечение данных из сессии (гонка или спринт)"""
        race_data = []
        session_type = 'sprint' if is_sprint else 'race'

        results = race_session.results

        for idx, row in results.iterrows():
            try:
                driver_abbr = row['Abbreviation']

                # Базовые данные
                data = {'year': year, 'event': event_name, 'session_type': session_type, 'driver': row['FullName'],
                        'driver_abbr': driver_abbr, 'team': self.normalize_team_name(row['TeamName']),
                        'position': row['Position'] if pd.notna(row['Position']) else 99,
                        'points': row['Points'] if pd.notna(row['Points']) else 0, 'status': row['Status'],
                        'grid_position': row['GridPosition'] if pd.notna(row['GridPosition']) else 20,
                        'dnf': 1 if 'DNF' in str(row['Status']) or 'Retired' in str(row['Status']) else 0,
                        'dsq': 1 if 'Disqualified' in str(row['Status']) else 0}

                # Квалификационные данные
                quali_data = self._get_quali_data(quali_session, driver_abbr)
                data.update(quali_data)

                # Данные гонки/спринта
                race_stats = self._get_race_stats(race_session, driver_abbr)
                data.update(race_stats)

                # Погода
                weather = self._get_weather_data(race_session)
                data.update(weather)

                race_data.append(data)

            except Exception:
                # Тихо пропускаем ошибки на уровне отдельных гонщиков
                continue

        return race_data

    @staticmethod
    def _get_quali_data(quali_session, driver_abbr):
        """Получение данных квалификации"""
        try:
            quali_results = quali_session.results
            driver_quali = quali_results[quali_results['Abbreviation'] == driver_abbr]

            if len(driver_quali) == 0:
                return {
                    'quali_position': 20,
                    'q1_time': np.nan,
                    'q2_time': np.nan,
                    'q3_time': np.nan,
                }

            driver_quali = driver_quali.iloc[0]

            return {
                'quali_position': driver_quali['Position'] if pd.notna(driver_quali['Position']) else 20,
                'q1_time': driver_quali['Q1'].total_seconds() if pd.notna(driver_quali['Q1']) else np.nan,
                'q2_time': driver_quali['Q2'].total_seconds() if pd.notna(driver_quali['Q2']) else np.nan,
                'q3_time': driver_quali['Q3'].total_seconds() if pd.notna(driver_quali['Q3']) else np.nan,
            }
        except Exception:
            return {
                'quali_position': 20,
                'q1_time': np.nan,
                'q2_time': np.nan,
                'q3_time': np.nan,
            }

    def _get_race_stats(self, race_session, driver_abbr):
        """Получение статистики гонки"""
        try:
            laps = race_session.laps.pick_drivers(driver_abbr)

            if len(laps) == 0:
                return self._empty_race_stats()

            # Статистика по кругам
            stats = {
                'total_laps': len(laps),
                'fastest_lap': laps['LapTime'].min().total_seconds() if not laps['LapTime'].isna().all() else np.nan,
                'avg_lap_time': laps['LapTime'].mean().total_seconds() if not laps['LapTime'].isna().all() else np.nan,
                'pit_stops': len(laps[laps['PitInTime'].notna()]),
            }

            # Телеметрия (средние значения)
            if 'SpeedST' in laps.columns:
                stats['avg_speed'] = laps['SpeedST'].mean() if not laps['SpeedST'].isna().all() else np.nan
            else:
                stats['avg_speed'] = np.nan

            return stats

        except Exception:
            return self._empty_race_stats()

    @staticmethod
    def _empty_race_stats():
        """Пустая статистика гонки"""
        return {
            'total_laps': 0,
            'fastest_lap': np.nan,
            'avg_lap_time': np.nan,
            'pit_stops': 0,
            'avg_speed': np.nan,
        }

    @staticmethod
    def _get_weather_data(race_session):
        """Получение данных о погоде"""
        try:
            weather = race_session.weather_data
            if weather is not None and len(weather) > 0:
                return {
                    'air_temp': weather['AirTemp'].mean(),
                    'track_temp': weather['TrackTemp'].mean(),
                    'humidity': weather['Humidity'].mean(),
                    'rainfall': weather['Rainfall'].any() * 1,
                }
        except Exception:
            pass

        return {
            'air_temp': np.nan,
            'track_temp': np.nan,
            'humidity': np.nan,
            'rainfall': 0,
        }

    def get_active_drivers_teams(self, year):
        """Получение списка активных гонщиков и команд"""
        try:
            schedule = fastf1.get_event_schedule(year)
            # Берем последнюю гонку сезона
            last_event = schedule.iloc[-1]['EventName']
            session = fastf1.get_session(year, last_event, 'R')
            session.load()

            results = session.results

            drivers = results['FullName'].tolist()
            teams = [self.normalize_team_name(t) for t in results['TeamName'].tolist()]

            return list(set(drivers)), list(set(teams))
        except Exception as e:
            print(f"⚠️  Не удалось получить список гонщиков и команд за {year}: {e}")
            return [], []
