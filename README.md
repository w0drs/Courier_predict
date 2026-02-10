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
    ├── data
    │   ├── external                            <- Data from third party sources.
    │   ├── interim                             <- Intermediate data that has been transformed.
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    ├── docs          
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks                               <- Jupyter notebooks
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports                                 <- Generated analysis
    ├── requirements.txt 
    ├── setup.py
    ├── src   
    │   ├── __init__.py
    │   ├── data                                <- Scripts to download or generate data
    │   │   ├── drop_features.py
    │   │   ├── merge_data.py
    │   │   ├── nan_filler.py
    │   │   └── nan_filling.py
    │   │
    │   ├── experiments                         <- Scripts with MlFlow experiments
    │   │   ├── baseline_experiment.py
    │   │   └── lgbm_with_params_experiment.py
    │   │
    │   ├── features                            <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                              <- Scripts to train models and then use trained models to make
    │   │   │                                      predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini 

--------

