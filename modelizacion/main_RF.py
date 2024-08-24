import pandas as pd
from joblib import Parallel, delayed
import mlflow
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from modelizacion.evaluar import score_rendimiento, get_tomorrow_rets
from modelizacion.train import train
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
        delayed(train)(name_exp, df, clf, param_grid, rmv_hg_corr, add_ft_sq, False, None, None, True, f_select,
                       scoring, max_features)
        for rmv_hg_corr in pruebas['rmv_hg_corr']
        for add_ft_sq in pruebas['add_ft_sq']
        for f_select in pruebas['f_select']
    )


if __name__ == '__main__':
    descarga_drive.main('archivos_final.txt')
    RANDOM_STATE = 42
    symbols = ['META']
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///dbs/RF1.db'
    for s in symbols:
        path = f'../../data/final_{s}.parquet'
        dataframe = pd.read_parquet(path)
        grid = {'criterion': ['gini', 'entropy'],
                'max_depth': [10, 25, 50],
                'min_samples_leaf': [1, 5, 10],
                'max_features': [10, 15, 20, 25, 30, 50, 75, 100, 'sqrt'],
                'n_estimators': [30, 50, 100]}
        params_train = {'rmv_hg_corr': [True, False],
                        'add_ft_sq': [True, False],
                        'f_select': ['kbest', 'rfe']}

        classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
        name_experiment = f'RF_{s}'
        mlflow.create_experiment(name_experiment)
        entrenar(name_experiment, classifier, dataframe, params_train, grid)
