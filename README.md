CourierPredict
==============================

## Проблема  
Сервисы доставки стремятся радовать своих клиентов и доставлять заказы за 15 мин. Курьеры - их движущая сила. Для того, чтобы обеспечить сервис высочайшего качества, необходимо прогнозировать будущий объем заказов, обеспечивать необходимое количество курьеров и оптимально планировать их индивидуальный рабочий график и смены.  

## Цель 
- Спрогнозировать для каждого места, в котором хранятся и собираются товары для торговли через интернет, будущий дефицит курьеров - количество курьеров, которое нужно нанять, чтобы в будущем доставлять все заказы без опозданий.  


## Структура проекта
--------
    CourierPredict/
    ├── LICENSE
    ├── Makefile
    ├── README.md 
    ├── data/
    │   ├── external/                           <- Data from third party sources.
    │   ├── interim/                            <- Intermediate data that has been transformed.
    │   ├── processed/                          <- The final, canonical data sets for modeling.
    │   └── raw/                                <- The original, immutable data dump.
    ├── docs/         
    ├── models/                                 <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks/                              <- Jupyter notebooks
    ├── references/                             <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports/                                <- Generated analysis
    ├── requirements.txt 
    ├── setup.py
    ├── src/
    │   ├── __init__.py
    │   ├── data/                               <- Scripts to download or generate data
    │   │   ├── drop_features.py
    │   │   ├── merge_data.py
    │   │   ├── nan_filler.py
    │   │   └── nan_filling.py
    │   │
    │   ├── experiments/                        <- Scripts with MlFlow experiments
    │   │   ├── baseline_experiment.py
    │   │   └── lgbm_with_params_experiment.py
    │   │
    │   ├── features/                           <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models/                             <- Scripts to train models and then use trained models to make
    │   │   │                                      predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization/                      <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tests/                                  <- Test data and train scripts
    │   ├── basic_test.py
    │   ├── test_drop_features.py
    │   ├── test_merge_data.py
    │   ├── test_nan_filler.py
    │   └── test_train_model.py
    ├── mlflow.db
    └── tox.ini 

--------

## Обучение модели
### Постановка задачи
Мы решаем задачу регрессии: прогнозирование дефицита курьеров для каждой торговой точки на неделю вперед.  

### Метрика оценки
WAPE (Weighted Absolute Percentage Error) - взвешенная абсолютная процентная ошибка. 
```text
WAPE = Σ|y_true - y_pred| / Σ|y_true|
```
Преимущества WAPE для этой задачи:
- Устойчива к выбросам  
- Интерпретируема  
- Не занижает ошибку при малых значениях (в отличие от MAPE)  \

| Модель | WAPE | Комментарий |
|--------|------|-------------|
| Базовое предсказание (среднее) | 0.9 | Прогноз средним значением |  
| LightGBM (без feature engineering) | 0.71 | Только исходные признаки |  
| LightGBM (с feature engineering) | 0.6 | Итоговая модель | 
Итог: улучшение на 35% относительно базового предсказания. 

### Признаки (Feature Engineering)
Внешние признаки:  
- Разница между прогнозом и реальностью прошлой недели.
- Сколько заказов на одного курьера в прошлой неделе.
- Прогнозная производительность на эту неделю.

Категориальные признаки:  
- Категориальные признаки обрабатывались самим lightgbm.

### Обработка пропусков  
Для новых магазинов (<7 дней работы):
- Числовые признаки заполняются 0.  
- Бинарные флаги заполняются 0.  
- Категориальные признаки заполняются 'пусто'.  

Для старых магазинов (≥7 дней):
- Числовые признаки заполняются медианой.  
- Бинарные флаги заполняются модой.  
- Категориальные признаки заполняются 'пусто'.  

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/w0drs/Courier_predict.git
cd CourierPredict

# Установка зависимостей через Poetry
poetry install

# Активация виртуального окружения
poetry shell
```

## Использование
### Подготовка данных
```bash
# Объединение данных
python src/data/merge_data.py data/raw/facts.csv data/raw/shifts.csv data/raw/train.csv data/raw/test.csv data/interim/train_merged.csv data/interim/test_merged.csv

# Удаление лишних признаков
python src/data/drop_features.py data/interim/train_merged.csv data/interim/test_merged.csv data/processed/train.csv data/processed/test.csv configs/params.yaml

# Заполнение пропусков
python src/data/nan_filling.py data/processed/train.csv data/processed/test.csv data/processed/train_filled.csv data/processed/test_filled.csv models/filler.joblib models/filler.joblib configs/params.yaml
```

### Обучение модели
```bash
python src/models/train_model.py data/processed/train_filled.csv models/model.joblib configs/params.yaml
```

### Предсказание
```bash
python src/models/predict_model.py models/model.joblib data/processed/test_filled.csv predictions/predictions.csv
```

### Эксперименты MlFlow
```bash
# Запуск эксперимента
python src/experiments/baseline_experiment.py

# Просмотр результатов
mlflow ui
```

### Тестирование
```bash
# Запуск всех тестов
pytest tests/

# С покрытием кода
pytest tests/ --cov=src/

# Конкретный модуль
pytest tests/test_nan_filler.py -v
```

