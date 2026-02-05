import pandas as pd
import click
import joblib
import yaml
from nan_filler import NanFiller
from pathlib import Path


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('train_output_path', type=click.Path())
@click.argument('test_output_path', type=click.Path())
@click.argument('filler_filepath', type=click.Path())
@click.argument('filler_output_path', type=click.Path())
@click.argument('params_file', type=click.Path())
def fill_nan(train_filepath,
             test_filepath,
             train_output_path,
             test_output_path,
             filler_filepath=None,
             filler_output_path=None,
             params_file='params.yaml') -> None:
    """Replaces empty values in the data by NanFiller
       Args:
           train_filepath: path to train data
           test_filepath: path to test data
           train_output_path: path where train data will be saved
           test_output_path: path where test data will be saved
           filler_filepath: path to empty values filler
           filler_output_path: path where filler will be saved if filler_filepath is empty or doesn't exist
           params_file: path to yaml file with parameters for this function (numeric, flag and cat columns)
        The final train and test data will be saved in csv files.
    """
    filler = None
    with open(params_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    numeric_cols = params['nan_filling']['numeric_cols']
    flag_cols = params['nan_filling']['flag_cols']
    cat_cols = params['nan_filling']['cat_cols']

    # Дальше ваш код без изменений
    train_df = pd.read_csv(train_filepath)

    if filler_filepath and Path(filler_filepath).exists():
        filler = joblib.load(filler_filepath)
        filled_df = filler.transform(train_df)
    else:
        filler = NanFiller()
        filler.fit(train_df, numeric_cols, flag_cols, cat_cols)
        filled_df = filler.transform(train_df)
        joblib.dump(filler, filler_output_path)

    # В датасете есть один объект, где 3 колонки с предсказаниями пусты. Этот один объект можем убрать
    cols_to_check = ['predicted_staff_value', 'predicted_num_orders', 'predicted_load_factor']
    filled_df = filled_df.dropna(subset=cols_to_check)

    if train_output_path:
        filled_df.to_csv(train_output_path, index=False)

    # Заполняем пропуски в test данных
    if test_filepath:
        test_df = pd.read_csv(test_filepath)
        test_filled = filler.transform(test_df)

        if test_output_path:
            test_filled.to_csv(test_output_path, index=False)


if __name__ == "__main__":
    fill_nan()
