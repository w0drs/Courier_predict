import pandas as pd
import click
import yaml
import joblib
import lightgbm as lgb
from pathlib import Path


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('model_output_path', type=click.Path())
@click.argument('config_path', type=click.Path(exists=True))
def train_lgbm_model(dataset_path, model_output_path, config_path):
    """
    Обучение LightGBM модели.

    Args:
        dataset_path: Путь к обучающему датасету (CSV)
        model_output_path: Путь для сохранения модели
        config_path: Путь к YAML файлу с параметрами модели
    """
    # Загрузка конфигурации из YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Извлечение параметров
    model_params = config.get('model_params', {})
    target_col = config.get('target_col', 'target')
    cat_features = config.get('categorical_features', [])


    # Загрузка данных
    df = pd.read_csv(dataset_path)

    # Проверка наличия таргета
    if target_col not in df.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в данных")

    # Подготовка данных
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    train_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_features
    )

    model = lgb.train(
        model_params,
        train_data,
        num_boost_round=2500,
    )

    joblib.dump(model, model_output_path, compress=3)

    # Сохранение информации о модели
    model_info = _get_model_info(
        model_params,
        dataset_path,
        X,
        y,
        target_col,
        cat_features
    )

    info_path = Path(model_output_path).with_suffix('.info.yaml')
    with open(info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)

    print(f"Модель сохранена: {model_output_path}")
    print(f"Информация о модели: {info_path}")
    print(f"Среднее целевой переменной: {y.mean():.2f}")

def _get_model_info(model_params,
                    dataset_path,
                    X_train,
                    y_train,
                    target_col,
                    cat_features) -> dict:
    model_info: dict = {
        'model': {
            'type': 'LightGBM',
            'params': model_params
        },
        'data': {
            'source': dataset_path,
            'shape': list(X_train.shape),
            'target': target_col,
            'categorical_features': cat_features
        },
        'metrics': {
            'target_mean': float(y_train.mean()),
            'target_std': float(y_train.std())
        },
        'training': {
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }

    return model_info


if __name__ == "__main__":
    train_lgbm_model()
