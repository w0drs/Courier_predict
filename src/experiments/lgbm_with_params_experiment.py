"""
Эксперимент: Baseline LightGBM модель с найденными параметрами.
Запускать: python -m src/experiments/lgbm_with_params_experiment.py
"""
import mlflow
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent
mlflow_path = project_root / "mlruns"
db_path = project_root / "mlflow.db"


def run_baseline_experiment():
    """Запуск baseline эксперимента"""
    print("Начало эксперимента")
    mlflow.set_tracking_uri(f"file:///{mlflow_path.absolute().as_posix()}")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("courier_deficit_baseline")
    print("Запуск...")
    with mlflow.start_run(run_name="lgbm_with_params_cv"):
        print("Загрузка данных")

        # Загрузка параметров модели, название target колонки и список категориальных фичей
        config_path = "../../configs/lgbm_parameters.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_params = config.get('model_params', {})
        target_col = config.get('target_col', 'target')
        cat_features = config.get('categorical_features', [])

        # Загрузка датасета
        dataset = pd.read_csv("../../data/processed/train_final.csv")

        # Разделение данных на колонки признаками и таргет колонку
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]

        # Преобразование категориальных признаков
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype('category')

        # Логируем параметры
        mlflow.log_params(model_params)

        # Кросс-валидация
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = {
            "mae": [],
            "rmse": [],
            "wape": []
        }
        print("Кроссвалидация")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Создание Dataset
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=cat_features
            )

            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                categorical_feature=cat_features
            )

            # Обучение
            model = lgb.train(
                params=model_params,
                train_set=train_data,
                num_boost_round=2500,
                valid_sets=[train_data, val_data],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.log_evaluation(500),
                    lgb.early_stopping(100)
                ]
            )

            # Предсказание и метрики
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            wape = mae / y_val.mean()

            # Сохраняем метрики фолда
            fold_metrics["mae"].append(mae)
            fold_metrics["rmse"].append(rmse)
            fold_metrics["wape"].append(wape)

            # Логируем метрики для каждого фолда
            mlflow.log_metric(f"fold_{fold}_mae", mae)
            mlflow.log_metric(f"fold_{fold}_rmse", rmse)
            mlflow.log_metric(f"fold_{fold}_wape", wape)
            mlflow.log_metric(f"fold_{fold}_best_iteration", model.best_iteration)

            print(f"Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}, WAPE={wape:.3f}")

        print("Сохранение метрик")

        # Средние метрики по всем фолдам
        mean_mae = np.mean(fold_metrics["mae"])
        mean_rmse = np.mean(fold_metrics["rmse"])
        mean_wape = np.mean(fold_metrics["wape"])

        std_mae = np.std(fold_metrics["mae"])
        std_rmse = np.std(fold_metrics["rmse"])
        std_wape = np.std(fold_metrics["wape"])

        # Логируем итоговые метрики
        mlflow.log_metrics({
            "cv_mean_mae": float(mean_mae),
            "cv_mean_rmse": float(mean_rmse),
            "cv_mean_wape": float(mean_wape),
            "cv_std_mae": float(std_mae),
            "cv_std_rmse": float(std_rmse),
            "cv_std_wape": float(std_wape)
        })

        print("Эксперимент проведен успешно!")


if __name__ == "__main__":
    run_baseline_experiment()
