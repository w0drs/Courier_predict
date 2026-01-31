import pandas as pd
import click
import yaml


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('train_output_path', type=click.Path())
@click.argument('test_output_path', type=click.Path())
@click.argument('params_file', type=click.Path())
def drop_features(train_filepath,
                  test_filepath,
                  train_output_path,
                  test_output_path,
                  params_file='params.yaml'):
    """Deleting unnecessary columns.
       Args:
           train_filepath: path to train data
           test_filepath: path to test data
           train_output_path: the path where train data will be saved
           test_output_path: the path where test data will be saved
           params_file: path to yaml file with parameters for this function (columns to drop)
       The final train and test data will be saved in csv files.
    """
    with open(params_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    drop_cols = params['drop_features']['cols']

    train_df = pd.read_csv(train_filepath)
    train_df.drop(columns=drop_cols, inplace=True)

    if train_output_path:
        train_df.to_csv(train_output_path, index=False)

    if test_filepath:
        test_df = pd.read_csv(test_filepath)
        test_df.drop(columns=drop_cols, inplace=True)
        if test_output_path:
            test_df.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    drop_features()