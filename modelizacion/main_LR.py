import pandas as pd
from joblib import Parallel, delayed
import mlflow
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from modelizacion.evaluar import score_rendimiento, get_tomorrow_rets
from modelizacion.train import train, get_scaler
import descarga_drive
import warnings
warnings.filterwarnings('ignore')


def entrenar(name_exp, clf, df, pruebas, param_grid, custom_scorer=True, max_features=30):
    """
        Entrena múltiples modelos utilizando diferentes configuraciones y realiza búsqueda en paralelo.

        Parámetros:
        - name_exp: Nombre del experimento de MLflow.
        - clf: Clasificador a entrenar.
        - df: DataFrame que contiene los datos.
        - pruebas: Diccionario con las combinaciones de parámetros a probar.
        - param_grid: Grilla de hiperparámetros a explorar.
        - custom_scorer: Booleano que indica si se utilizará un scorer personalizado (por defecto True).
        - max_features: Número máximo de características a considerar (por defecto 30).

        Devuelve:
        - No devuelve ningún valor, pero entrena y guarda múltiples modelos en paralelo utilizando MLflow.
    """
    tomorrow_rets = get_tomorrow_rets(df)
    scoring = 'balanced_accuracy' if not custom_scorer else make_scorer(score_rendimiento, tomorrow_rets=tomorrow_rets)
    Parallel(n_jobs=-1, verbose=20)(
        delayed(train)(name_exp, df, clf, param_grid, rmv_hg_corr, False, True, get_scaler(scaler), cols, True, 'rfe',
                       scoring, max_features)
        for rmv_hg_corr in pruebas['rmv_hg_corr']
        for scaler in pruebas['scaler']
        for cols in pruebas['cols_to_scale']
    )


if __name__ == '__main__':
    descarga_drive.main('archivos_final.txt')
    RANDOM_STATE = 42
    symbols = ['AAPL', 'F', 'GOOG', 'META', 'MSFT', 'TSLA']
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///dbs/LR1.db'
    for s in symbols:
        path = f'../../data/final_{s}.parquet'
        dataframe = pd.read_parquet(path)
        grid = {'C': [.001, .01, .1, 1, 10],
                'penalty': [None, 'l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']}
        params_train = {'rmv_hg_corr': [True, False],
                        'scaler': ['standard', 'minmax', 'robust', 'yeo-johnson'],
                        'cols_to_scale': ['continuous', 'big']}

        classifier = LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
        name_experiment = f'LR_{s}'
        mlflow.create_experiment(name_experiment)
        entrenar(name_experiment, classifier, dataframe, params_train, grid)
