"""
Визуализация результатов (улучшенная версия)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ResultsPlotter:
    def __init__(self):
        """Инициализация плоттера"""
        self.figsize = (14, 8)

    def plot_test_results(self, actual_drivers, pred_drivers, actual_teams, pred_teams, output_dir):
        """Визуализация результатов тестирования"""
        output_dir = Path(output_dir)

        # 1. Сравнение топ-10 гонщиков
        self._plot_drivers_comparison(actual_drivers, pred_drivers, output_dir)

        # 2. Сравнение команд
        self._plot_teams_comparison(actual_teams, pred_teams, output_dir)

        # 3. Scatter plot: предсказанные vs реальные очки
        self._plot_prediction_scatter(actual_drivers, pred_drivers, output_dir)

        print(f"✅ Визуализации сохранены в {output_dir}")

    def plot_predictions(self, driver_results, team_results, output_dir, year):
        """Визуализация предсказаний на будущее"""
        output_dir = Path(output_dir)

        # 1. Топ-10 гонщиков с детальной статистикой
        self._plot_top_drivers_prediction(driver_results, output_dir, year)

        # 2. Все команды
        self._plot_teams_prediction(team_results, output_dir, year)

        # 3. Распределение очков
        self._plot_points_distribution(driver_results, output_dir, year)

        # 4. Статистика побед и подиумов
        if 'wins' in driver_results.columns:
            self._plot_wins_podiums_stats(driver_results, output_dir, year)

        # 5. Сравнение команд детально
        self._plot_team_comparison_detailed(team_results, output_dir, year)

        print(f"✅ Визуализации сохранены в {output_dir}")

    @staticmethod
    def _plot_drivers_comparison(actual, predicted, output_dir):
        """Сравнение реальных и предсказанных результатов гонщиков"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Берем топ-10
        actual_top10 = actual.head(10)
        pred_top10 = predicted.head(10)

        # График 1: Реальные результаты
        ax1.barh(range(len(actual_top10)), actual_top10['total_points'], color='steelblue')
        ax1.set_yticks(range(len(actual_top10)))
        ax1.set_yticklabels(actual_top10['driver'], fontsize=10)
        ax1.set_xlabel('Очки', fontsize=12)
        ax1.set_title('Реальные результаты - Топ-10 гонщиков', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()

        # Добавляем значения
        for i, v in enumerate(actual_top10['total_points']):
            ax1.text(v + 5, i, f'{v:.0f}', va='center', fontsize=10)

        # График 2: Предсказанные результаты
        ax2.barh(range(len(pred_top10)), pred_top10['total_points'], color='coral')
        ax2.set_yticks(range(len(pred_top10)))
        ax2.set_yticklabels(pred_top10['driver'], fontsize=10)
        ax2.set_xlabel('Очки', fontsize=12)
        ax2.set_title('Предсказанные результаты - Топ-10 гонщиков', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()

        for i, v in enumerate(pred_top10['total_points']):
            ax2.text(v + 5, i, f'{v:.0f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'drivers_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_teams_comparison(self, actual, predicted, output_dir):
        """Сравнение результатов команд"""
        fig, ax = plt.subplots(figsize=self.figsize)

        teams = actual['team'].values
        x = np.arange(len(teams))
        width = 0.35

        ax.bar(x - width/2, actual['total_points'], width, label='Реальные', color='steelblue')
        ax.bar(x + width/2, predicted['total_points'], width, label='Предсказанные', color='coral')

        ax.set_xlabel('Команда', fontsize=12)
        ax.set_ylabel('Очки', fontsize=12)
        ax.set_title('Сравнение результатов команд', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(teams, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'teams_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_prediction_scatter(actual, predicted, output_dir):
        """Scatter plot реальных vs предсказанных очков"""
        # Объединяем данные по гонщикам
        merged = actual[['driver', 'total_points']].merge(
            predicted[['driver', 'total_points']],
            on='driver',
            suffixes=('_actual', '_predicted')
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(merged['total_points_actual'], merged['total_points_predicted'],
                  s=100, alpha=0.6, color='mediumpurple')

        # Линия идеального предсказания
        max_val = max(merged['total_points_actual'].max(), merged['total_points_predicted'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеальное предсказание')

        ax.set_xlabel('Реальные очки', fontsize=12)
        ax.set_ylabel('Предсказанные очки', fontsize=12)
        ax.set_title('Точность предсказаний (Реальные vs Предсказанные)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_top_drivers_prediction(driver_results, output_dir, year):
        """График топ-10 предсказанных гонщиков с улучшенной визуализацией"""
        fig, ax = plt.subplots(figsize=(16, 10))

        top10 = driver_results.head(10)

        y_pos = np.arange(len(top10))

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['driver']}\n({row['team']})"
                            for _, row in top10.iterrows()], fontsize=11)
        ax.set_xlabel('Предсказанные очки', fontsize=13, fontweight='bold')
        ax.set_title(f'Предсказание топ-10 гонщиков - сезон {year}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Добавляем детальную информацию
        for i, (_, row) in enumerate(top10.iterrows()):
            points = row['total_points']
            ax.text(points + 15, i, f"{points:.0f}",
                   va='center', fontsize=11, fontweight='bold')

            # Если есть статистика побед/подиумов
            if 'wins' in row and 'podiums' in row:
                stats_text = f"W:{int(row['wins'])} P:{int(row['podiums'])}"
                ax.text(points/2, i, stats_text,
                       va='center', ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'top10_drivers_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_teams_prediction(team_results, output_dir, year):
        """График предсказаний для всех команд"""
        fig, ax = plt.subplots(figsize=(16, 9))

        ax.set_xticks(range(len(team_results)))
        ax.set_xticklabels(team_results['team'], rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Предсказанные очки', fontsize=13, fontweight='bold')
        ax.set_title(f'F1 | Предсказание результатов команд - сезон {year}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Добавляем значения и статистику
        for i, (_, row) in enumerate(team_results.iterrows()):
            points = row['total_points']
            ax.text(i, points + 20, f"{points:.0f}",
                   ha='center', fontsize=10, fontweight='bold')

            # Если есть статистика побед
            if 'wins' in row and row['wins'] > 0:
                ax.text(i, points/2, f"Wins: {int(row['wins'])}",
                       ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

        plt.tight_layout()
        plt.savefig(output_dir / f'teams_prediction_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_points_distribution(driver_results, output_dir, year):
        """Распределение очков среди гонщиков"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Гистограмма распределения очков
        ax1.hist(driver_results['total_points'], bins=15,
                color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(driver_results['total_points'].median(),
                   color='red', linestyle='--', linewidth=2,
                   label=f"Медиана: {driver_results['total_points'].median():.0f}")
        ax1.set_xlabel('Очки', fontsize=12)
        ax1.set_ylabel('Количество гонщиков', fontsize=12)
        ax1.set_title(f'Распределение очков - сезон {year}',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Box plot средних позиций по командам
        team_data = []
        team_labels = []

        for team in driver_results['team'].unique():
            team_drivers = driver_results[driver_results['team'] == team]
            if len(team_drivers) > 0:
                team_data.append(team_drivers['avg_position'].values)
                team_labels.append(team)

        bp = ax2.boxplot(team_data, labels=team_labels, patch_artist=True)

        # Раскрашиваем box plots
        colors = plt.cm.Set3(np.linspace(0, 1, len(team_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax2.set_xticklabels(team_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Средняя позиция', fontsize=12)
        ax2.set_title(f'Распределение позиций по командам - сезон {year}',
                     fontsize=13, fontweight='bold')
        ax2.invert_yaxis()  # Меньшая позиция = лучше
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'points_distribution_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_wins_podiums_stats(driver_results, output_dir, year):
        """Статистика побед и подиумов"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Фильтруем только гонщиков с подиумами
        podium_drivers = driver_results[driver_results['podiums'] > 0].head(10)

        if len(podium_drivers) > 0:
            # График 1: Победы vs Подиумы
            x = np.arange(len(podium_drivers))
            width = 0.35

            ax1.bar(x - width/2, podium_drivers['wins'], width,
                   label='Победы', color='gold', edgecolor='black')
            ax1.bar(x + width/2, podium_drivers['podiums'], width,
                   label='Подиумы', color='silver', edgecolor='black')

            ax1.set_xlabel('Гонщик', fontsize=12)
            ax1.set_ylabel('Количество', fontsize=12)
            ax1.set_title(f'Победы и подиумы - сезон {year}',
                         fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(podium_drivers['driver'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

        # График 2: Процент финишей в очках
        if 'points_finishes' in driver_results.columns:
            top10 = driver_results.head(10)
            points_finish_rate = (top10['points_finishes'] / 24) * 100  # 24 гонки

            ax2.set_yticks(range(len(top10)))
            ax2.set_yticklabels(top10['driver'])
            ax2.set_xlabel('Процент финишей в очках (%)', fontsize=12)
            ax2.set_title(f'Стабильность гонщиков - сезон {year}',
                         fontsize=14, fontweight='bold')
            ax2.invert_yaxis()

            # Добавляем значения
            for i, v in enumerate(points_finish_rate):
                ax2.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=10)

            ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'wins_podiums_stats_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_team_comparison_detailed(team_results, output_dir, year):
        """Детальное сравнение команд"""
        fig, ax = plt.subplots(figsize=(14, 8))

        top_teams = team_results.head(5)

        x = np.arange(len(top_teams))

        # Создаем составной график
        ax.bar(x, top_teams['total_points'], color='steelblue',
              edgecolor='black', linewidth=1.5, alpha=0.7, label='Общие очки')

        ax.set_xlabel('Команда', fontsize=13, fontweight='bold')
        ax.set_ylabel('Очки', fontsize=13, fontweight='bold')
        ax.set_title(f'F1 | Топ-5 команд детально - сезон {year}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(top_teams['team'], rotation=30, ha='right', fontsize=11)

        # Добавляем детальную информацию
        for i, (_, row) in enumerate(top_teams.iterrows()):
            points = row['total_points']

            # Основные очки
            ax.text(i, points + 20, f"{points:.0f}",
                   ha='center', fontsize=12, fontweight='bold')

            # Статистика
            info_text = f"Avg pos: {row['avg_position']:.1f}"
            if 'wins' in row:
                info_text += f"\nWins: {int(row['wins'])}"
            if 'podiums' in row:
                info_text += f"\nPodiums: {int(row['podiums'])}"

            ax.text(i, points/2, info_text,
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='black', alpha=0.8))

        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(output_dir / f'team_comparison_detailed_{year}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()