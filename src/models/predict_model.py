import pandas as pd
import click
import joblib


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def predict_model(model_path, data_path, output_path):
    """
    Предсказание с использованием обученной LightGBM модели.
    Предполагается, что данные уже готовы для модели.

    Args:
        model_path: Путь к сохранённой модели (.joblib)
        data_path: Путь к готовым данным для предсказания (CSV)
        output_path: Путь для сохранения предсказаний (CSV)
    """
    model = joblib.load(model_path)

    X = pd.read_csv(data_path)

    predictions = model.predict(X)

    pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)


if __name__ == "__main__":
    predict_model()
