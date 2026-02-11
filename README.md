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


## Модель и результаты
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
- Не занижает ошибку при малых значениях (в отличие от MAPE)

### Модель 
Для задачи прогнозирования дефицита курьеров был выбран LightGBM по трем причинам:
- Скорость - обучение в 2-3 раза быстрее XGBoost  
- Категориальные признаки - нативная поддержка без one-hot encoding  
- Память - нужно меньше RAM, чем у аналогов

Результат: WAPE = 0.63 при обучении за ~20 секунд - оптимальный баланс качества и скорости.  

### Результаты различных моделей  
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


## Воспроизводимость с DVC
### Почему DVC?
Проект использует DVC (Data Version Control) для версионирования данных и воспроизводимости экспериментов.  
Каждый этап обработки зафиксирован в пайплайне, что гарантирует:  
- Воспроизводимость - любой запуск дает тот же результат
- Версионирование данных - данные привязаны к коммитам
- Параллельные эксперименты - легкое переключение между версиями

### Пайплайн обработки данных  
```bash
# Воспроизвести весь пайплайн
dvc repro

# Визуализировать пайплайн
dvc dag
```

### Этапы пайплайна  
| Этап | Входные данные | Выходные данные | Описание |
|------|----------------|-----------------|----------|
| merge-data | facts, shifts, train, test | train_merged, test_merged | Объединение данных за разные недели |  
| fill-nan | train_merged, test_merged | train_filled, test_filled, nan_filler | Заполнение пропусков (кастомный NanFiller) |  
| build-features | train_filled, test_filled | train_features, test_features | Генерация признаков (лаги, скользящие средние) |  
| drop-features | train_features, test_features | train_final, test_final | Удаление неинформативных признаков |   

### Версионирование данных
```bash
# Скачать данные из хранилища (gdrive хранилище)
dvc pull

# Посмотреть различия в данных
dvc diff

# Переключиться на другую версию данных
git checkout <commit>
dvc checkout
```

## Эксперименты и логирование (MLflow)
```bash
# Запустить эксперимент
python src/experiments/baseline_experiment.py

# Запустить эксперимент с подбранными параметрами
python src/experiments/lgbm_with_params_experiment.py

# Просмотр результатов
mlflow ui  # http://localhost:5000
```

Логируется:
- Параметры модели
- Инфрмация о датасете  
- WAPE, RMSE, MAE  

## Тестирование
```bash
# Запуск всех тестов
pytest tests/

# Конкретный модуль
pytest tests/test_nan_filler.py -v
```

## CI/CD: GitHub Actions  
### Автоматическая проверка кода  
Каждый push в ветки master/main и каждый pull request запускает CI-пайплайн, который автоматически проверяет качество кода и работоспособность модели.  

### Что проверяется:
| Этап | Инструмент | Что проверяет |
|--------|------|-------------|
| Кодстайл | `flake8` | Синтаксические ошибки, неиспользуемые импорты |  
| Тесты | `pytest` | Корректность работы функций |  

### Преимущества для проекта  
- Качество кода — автоматический контроль стандартов  
- Защита от регрессий — тесты не дают сломать существующую логику  
- Экономия времени — код проверяется без участия разработчика

### Локальный запуск
Те же проверки можно выполнить локально:  
```bash
# Линтинг
poetry run flake8 src/ tests/

# Тесты
poetry run pytest tests/
```
