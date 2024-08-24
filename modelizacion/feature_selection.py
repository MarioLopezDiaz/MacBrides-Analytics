import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SequentialFeatureSelector, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from modelizacion.evaluar import score_rendimiento, get_tomorrow_rets
from modelizacion.preparacion import preparar_datos
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def my_mutual_info_classif(x, y):
    """
        Calcula la importancia de las características utilizando el método mutual_info_classif con un random_state fijo.

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.

        Devuelve:
        - Array con la importancia de cada característica.
    """
    return mutual_info_classif(x, y, random_state=RANDOM_STATE)


def select_kbest(x, y, k, score_func=my_mutual_info_classif):
    """
        Selecciona las k mejores características utilizando SelectKBest.

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - k: Número de características a seleccionar.
        - score_func: Función de puntuación para evaluar la importancia de las características
                      (por defecto my_mutual_info_classif).

        Devuelve:
        - DataFrame con las k mejores características seleccionadas.
    """
    selector = SelectKBest(k=k, score_func=score_func)
    x_selected = selector.fit_transform(x, y)
    selected_column_names = x.columns[selector.get_support(indices=True)]
    return pd.DataFrame(x_selected, columns=selected_column_names)


def sequential_selection(x, y, k, clf, scoring='balanced_accuracy', direction='forward'):
    """
        Realiza una selección secuencial de características utilizando SequentialFeatureSelector.

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - k: Número de características a seleccionar.
        - clf: Estimador utilizado para evaluar las características.
        - scoring: Métrica de evaluación (por defecto 'balanced_accuracy').
        - direction: Dirección de la selección ('forward' o 'backward', por defecto 'forward').

        Devuelve:
        - DataFrame con las características seleccionadas.
    """
    sfs = SequentialFeatureSelector(clf, n_features_to_select=k, direction=direction, scoring=scoring, n_jobs=-1)
    sfs.fit(x, y)
    selected_columns = x.columns[sfs.get_support()]
    return pd.DataFrame(sfs.transform(x), columns=selected_columns)


def selection_rfe(x, y, k, clf):
    """
        Realiza una selección de características utilizando Recursive Feature Elimination (RFE).

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - k: Número de características a seleccionar.
        - clf: Estimador utilizado para evaluar las características.

        Devuelve:
        - DataFrame con las características seleccionadas.
    """
    rfe = RFE(estimator=clf, n_features_to_select=k, step=2)
    rfe.fit(x, y)
    selected_columns = x.columns[rfe.get_support()]
    return pd.DataFrame(rfe.transform(x), columns=selected_columns)


def feature_selection(method, x, y, k, clf=None, sckbest=my_mutual_info_classif, scoring='balanced_accuracy',
                      dirss='forward'):
    """
        Realiza la selección de características utilizando el método especificado.

        Parámetros:
        - method: Método de selección de características ('kbest', 'sequential' o 'rfe').
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - k: Número de características a seleccionar.
        - clf: Estimador utilizado para métodos que lo requieran (por defecto None).
        - sckbest: Función de puntuación para SelectKBest (por defecto my_mutual_info_classif).
        - scoring: Métrica de evaluación para SequentialFeatureSelector (por defecto 'balanced_accuracy').
        - dirss: Dirección de selección para SequentialFeatureSelector ('forward' o 'backward', por defecto 'forward').

        Devuelve:
        - DataFrame con las características seleccionadas.
        """
    if method == 'kbest':
        return select_kbest(x, y, k, sckbest)
    if method == 'sequential':
        return sequential_selection(x, y, k, clf, scoring, dirss)
    if method == 'rfe':
        return selection_rfe(x, y, k, clf)
    raise ValueError("""Método no válido. Los métodos válidos son: 'kbest', 'sequential', 'rfe'""")


if __name__ == '__main__':
    dataframe = pd.read_parquet('../data/final_AAPL.parquet')
    tomorrow_rets = get_tomorrow_rets(dataframe)
    custom_score = make_scorer(score_rendimiento, tomorrow_rets=tomorrow_rets)
    X_train, X_test, y_train, y_test = preparar_datos(dataframe, rmv_hg_corr=False)
    classifier = LogisticRegression(max_iter=500)
    print(feature_selection('kbest', X_train, y_train, 30))
    print(feature_selection('sequential', X_train, y_train, 30, classifier, scoring=custom_score))
    print(feature_selection('rfe', X_train, y_train, 30, classifier))
