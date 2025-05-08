import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import warnings

# Подавляем предупреждения
warnings.filterwarnings("ignore")

def main():
    # --- ЭТАП 1. Загрузка и анализ данных ---
    train_df = pd.read_csv(r"C:\Users\user\Downloads\upload\train.csv")
    # Для конкурсных решений отладочный вывод лучше убрать, чтобы не смешивать результат с форматом файла.
    # Например, можно раскомментировать при локальном тестировании:
    # print("Размер обучающего набора:", train_df.shape)
    
    X = train_df.drop('y', axis=1)
    y = train_df['y']
    
    # --- ЭТАП 2. Подбор гиперпараметров через grid search ---
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [500, 1000, 1500]
    }
    
    # Инициализируем базовую модель
    base_model = CatBoostRegressor(
        loss_function='MAE',
        random_seed=42,
        verbose=50  # Можно отключить вывод при финальном запуске
    )
    
    fit_params = {"early_stopping_rounds": 50}
    
    # Пробуем выполнить grid search для подбора гиперпараметров.
    # Если возникает исключение, переходим к значениям по умолчанию.
    try:
        grid_results = base_model.grid_search(param_grid, X, y,
                                              cv=3,
                                              partition_random_seed=42,
                                              fit_params=fit_params)
        best_iterations = grid_results.get('iterations', 1000)
        best_depth = grid_results.get('depth', 6)
        best_learning_rate = grid_results.get('learning_rate', 0.05)
    except Exception as e:
        # Отладочный вывод можно оставить при локальном тестировании
        # print("Исключение в grid_search:", e)
        best_iterations = 1000
        best_depth = 6
        best_learning_rate = 0.05

    # --- ЭТАП 3. Ансамблирование моделей ---
    # Обучаем две модели с подобранными (или значениями по умолчанию) гиперпараметрами и различными random_seed
    model1 = CatBoostRegressor(
        iterations=best_iterations,
        depth=best_depth,
        learning_rate=best_learning_rate,
        loss_function='MAE',
        random_seed=42,
        verbose=False
    )
    model2 = CatBoostRegressor(
        iterations=best_iterations,
        depth=best_depth,
        learning_rate=best_learning_rate,
        loss_function='MAE',
        random_seed=43,
        verbose=False
    )
    
    model1.fit(X, y, verbose=False)
    model2.fit(X, y, verbose=False)
    
    # --- ЭТАП 4. Предсказание на тестовом наборе и сохранение результатов ---
    X_test = pd.read_csv('test_x.csv')
    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)
    
    # Усредняем предсказания обеих моделей
    ensemble_preds = (preds1 + preds2) / 2
    
    # Сохраняем результат в файл test_y.csv с заголовком 'y'
    pd.Series(ensemble_preds, name='y').to_csv('test_y.csv', index=False)

if __name__ == '__main__':
    main()
