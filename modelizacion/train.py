from datetime import datetime
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from modelizacion.evaluar import get_tomorrow_rets, evaluar
from modelizacion.feature_selection import my_mutual_info_classif, feature_selection
from modelizacion.preparacion import preparar_datos
import warnings
warnings.filterwarnings('ignore')


def get_scaler(scaler):
    if scaler == 'minmax':
        return MinMaxScaler()
    elif scaler == 'standard':
        return StandardScaler()
    elif scaler == 'robust':
        return RobustScaler()
    elif scaler == 'yeo-johnson':
        return PowerTransformer()
    raise


def cv_selection(n, train_size=.8):
    """
        Genera una lista de índices para realizar una validación cruzada simple.

        Parámetros:
        - n: Número total de muestras.
        - train_size: Proporción del conjunto de datos que se utilizará como conjunto de entrenamiento (por defecto 0.8)

        Devuelve:
        - Lista de tuplas de índices de entrenamiento y prueba.
    """
    train_indices = list(range(int(n * train_size)))
    test_indices = list(range(int(n * train_size), n))
    return [(train_indices, test_indices)]


def grid_search_train(x, y, clf, param_grid, train_size=.8, scoring='balanced_accuracy'):
    """
        Realiza una búsqueda en rejilla (grid search) para entrenar un clasificador con validación cruzada simple.

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - clf: Clasificador a entrenar.
        - param_grid: Rejilla de hiperparámetros a explorar.
        - train_size: Proporción del conjunto de datos que se utilizará como conjunto de entrenamiento (por defecto 0.8)
        - scoring: Métrica de evaluación (por defecto 'balanced_accuracy').

        Devuelve:
        - grid_search: Objeto GridSearchCV entrenado.
    """
    cv = cv_selection(len(x), train_size)
    grid_search = GridSearchCV(clf, param_grid, scoring=scoring, cv=cv)
    grid_search.fit(x, y)
    return grid_search


def train_model(x, y, clf, param_grid, do_feat_select=True, f_selection='kbest', sckbest=my_mutual_info_classif,
                dirss='forward', train_size=.8, scgs='balanced_accuracy', max_features=30):
    """
        Entrena un modelo utilizando un clasificador y busca el número de variables óptimo, las variables óptimas
        y los hiperparámetros óptimos.

        Parámetros:
        - x: Datos de entrada.
        - y: Etiquetas de clase.
        - clf: Clasificador a entrenar.
        - param_grid: Rejilla de hiperparámetros a explorar.
        - do_feat_select: Booleano que indica si se debe realizar selección de características (por defecto True).
        - f_selection: Método de selección de características ('kbest', 'sequential' o 'rfe', por defecto 'kbest').
        - sckbest: Función de puntuación para SelectKBest (por defecto my_mutual_info_classif).
        - dirss: Dirección de selección para SequentialFeatureSelector ('forward' o 'backward', por defecto 'forward').
        - train_size: Proporción del conjunto de datos que se utilizará como conjunto de entrenamiento (por defecto 0.8)
        - scgs: Métrica de evaluación para GridSearchCV (por defecto 'balanced_accuracy').
        - max_features: Número máximo de características a considerar (por defecto 30).

        Devuelve:
        - best_model: Mejor modelo entrenado. Es un objeto GridSearchCV que contiene información
        sobre la búsquedany sus resultados
    """
    best_model = None
    for k in range(1, max_features):
        x_new = x
        if do_feat_select:
            x_new = feature_selection(f_selection, x, y, k, clf, sckbest, dirss)
        m = grid_search_train(x_new, y, clf, param_grid, train_size=train_size, scoring=scgs)
        if best_model is None or m.best_score_ > best_model.best_score_:
            best_model = m
        if not do_feat_select:
            break
    return best_model


def guardar_modelo_mlflow(name_experiment, modelo, x_train, x_test, y_train, y_test, tmrw_rets, rmv_hg_corr, add_ft_sq,
                          trans_data, scaler, cols_to_scale, do_feat_sel, f_select, scoring, train_size):
    """
        Guarda el modelo y sus parámetros en MLflow, junto con métricas de evaluación.

        Parámetros:
        - name_experiment: Nombre del experimento de MLflow.
        - modelo: Mejor modelo entrenado.
        - x_train: Datos de entrenamiento.
        - x_test: Datos de prueba.
        - y_train: Etiquetas de entrenamiento.
        - y_test: Etiquetas de prueba.
        - tmrw_rets: Rendimientos de las acciones para el día siguiente.
        - rmv_hg_corr: Booleano que indica si se eliminaron las características con alta correlación.
        - add_ft_sq: Booleano que indica si se agregaron características al cuadrado.
        - trans_data: Booleano que indica si se realizaron transformaciones de datos.
        - scaler: Escalador utilizado.
        - cols_to_scale: Columnas escaladas.
        - do_feat_sel: Booleano que indica si se realizó selección de características.
        - f_select: Método de selección de características.
        - scoring: Métrica de evaluación.
        - train_size: Proporción de datos utilizada para entrenamiento.

        Devuelve:
        - No devuelve ningún valor, pero guarda el modelo y sus parámetros en MLflow.
    """
    mlflow.set_experiment(name_experiment)
    run_name = f'{rmv_hg_corr}_{add_ft_sq}_{trans_data}_{scaler}_{cols_to_scale}_{do_feat_sel}_{f_select}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('Remove_corr', rmv_hg_corr)
        mlflow.log_param('Add_feat_sqr', add_ft_sq)
        mlflow.log_param('Trans_data', trans_data)
        mlflow.log_param('Scaler', scaler)
        mlflow.log_param('Cols_to_scale', cols_to_scale)
        mlflow.log_param('Do_feat_select', do_feat_sel)
        mlflow.log_param('f_select', f_select)
        mlflow.log_param('scoring', scoring if scoring == 'balanced_accuracy' else 'custom_scorer')
        for k, v in modelo.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_param('Num_cols', len(modelo.feature_names_in_))
        mlflow.log_param('Cols', modelo.feature_names_in_)
        mlflow.log_metric('Score', modelo.best_score_)

        (av, bav, rev, pv, rsv), (at, bat, ret, pt, rst) = evaluar(modelo.best_estimator_,
                                                                   x_train[modelo.feature_names_in_],
                                                                   x_test[modelo.feature_names_in_],
                                                                   y_train, y_test, tmrw_rets, train_size=train_size)
        mlflow.log_metric('Accuracy_validation', av)
        mlflow.log_metric('Balanced_accuracy_validation', bav)
        mlflow.log_metric('Recall_validation', rev)
        mlflow.log_metric('Precision_validation', pv)
        mlflow.log_metric('Rendimiento_stock_validation', rsv)

        mlflow.log_metric('Accuracy_test', at)
        mlflow.log_metric('Balanced_accuracy_test', bat)
        mlflow.log_metric('Recall_test', ret)
        mlflow.log_metric('Precision_test', pt)
        mlflow.log_metric('Rendimiento_stock_test', rst)
        mlflow.sklearn.log_model(modelo.best_estimator_, f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def get_best_model(name_experiment, df, clf, param_grid, rmv_hg_corr=True, add_ft_sq=True, trans_data=True,
                   scaler=StandardScaler(), columns_to_scale='continuous', do_feat_select=True, f_select='kbest',
                   dirss='forward', sckbest=my_mutual_info_classif, train_size=.8, scgs='balanced_accuracy',
                   max_features=30):
    """
        Obtiene el mejor modelo utilizando preparación de datos, entrenamiento de modelo y lo guarda en MLflow.

        Parámetros:
        - name_experiment: Nombre del experimento de MLflow.
        - df: DataFrame que contiene los datos.
        - clf: Clasificador a entrenar.
        - param_grid: Rejilla de hiperparámetros a explorar.
        - rmv_hg_corr: Booleano que indica si se deben eliminar las características con alta correlación
        - add_ft_sq: Booleano que indica si se deben agregar características al cuadrado (por defecto True).
        - trans_data: Booleano que indica si se deben transformar los datos (por defecto True).
        - scaler: Escalador utilizado (por defecto StandardScaler()).
        - columns_to_scale: Tipo de columnas a escalar ('continuous', 'all', 'big' o None) (por defecto 'continuous').
        - do_feat_select: Booleano que indica si se debe realizar selección de características (por defecto True).
        - f_select: Método de selección de características ('kbest', 'sequential' o 'rfe', por defecto 'kbest').
        - dirss: Dirección de selección para SequentialFeatureSelector ('forward' o 'backward', por defecto 'forward').
        - sckbest: Función de puntuación para SelectKBest (por defecto my_mutual_info_classif).
        - train_size: Proporción del conjunto de datos que se utilizará como conjunto de entrenamiento (por defecto 0.8)
        - scgs: Métrica de evaluación para GridSearchCV (por defecto 'balanced_accuracy').
        - max_features: Número máximo de características a considerar (por defecto 30).

        Devuelve:
        - best_model: Mejor modelo entrenado.
    """
    x_train, x_test, y_train, y_test = preparar_datos(df, rmv_hg_corr, add_ft_sq, trans_data, scaler, columns_to_scale)
    best_model = train_model(x_train, y_train, clf, param_grid, do_feat_select=do_feat_select, f_selection=f_select,
                             sckbest=sckbest, dirss=dirss, train_size=train_size, scgs=scgs, max_features=max_features)
    tmrw_rets = get_tomorrow_rets(df)
    guardar_modelo_mlflow(name_experiment, best_model, x_train, x_test, y_train, y_test, tmrw_rets, rmv_hg_corr,
                          add_ft_sq, trans_data, scaler, columns_to_scale, do_feat_select, f_select, scgs, train_size)
    return best_model


def train(name_experiment, df, clf, param_grid, rmv_hg_corr=True, add_ft_sq=True, trans_data=True,
          scaler=StandardScaler(), cols_to_scale='continuous', do_feat_sel=True, f_select='kbest',
          scoring='balanced_accuracy', max_features=30):
    """
        Entrena un modelo utilizando un clasificador dado, realizando preparación de datos, selección de características
        y búsqueda en rejilla.

        Parámetros:
        - name_experiment: Nombre del experimento de MLflow.
        - df: DataFrame que contiene los datos.
        - clf: Clasificador a entrenar.
        - param_grid: Rejilla de hiperparámetros a explorar.
        - rmv_hg_corr: Booleano que indica si se deben eliminar las características con alta correlación
        - add_ft_sq: Booleano que indica si se deben agregar características al cuadrado (por defecto True).
        - trans_data: Booleano que indica si se deben transformar los datos (por defecto True).
        - scaler: Escalador utilizado (por defecto StandardScaler()).
        - cols_to_scale: Tipo de columnas a escalar ('continuous', 'all', 'big' o None) (por defecto 'continuous').
        - do_feat_sel: Booleano que indica si se debe realizar selección de características (por defecto True).
        - f_select: Método de selección de características ('kbest', 'sequential' o 'rfe', por defecto 'kbest').
        - scoring: Métrica de evaluación para GridSearchCV (por defecto 'balanced_accuracy').
        - max_features: Número máximo de características a considerar (por defecto 30).

        Devuelve:
        - best_model: Mejor modelo entrenado.
    """
    return get_best_model(name_experiment, df, clf, param_grid, rmv_hg_corr=rmv_hg_corr, add_ft_sq=add_ft_sq,
                          trans_data=trans_data, scaler=scaler, columns_to_scale=cols_to_scale,
                          do_feat_select=do_feat_sel, f_select=f_select, scgs=scoring, max_features=max_features)


if __name__ == '__main__':
    RANDOM_STATE = 42
    dataframe = pd.read_parquet('../data/final_AAPL.parquet')
    clasifier = LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    grid = {'C': [0.001, 0.01, 0.1, 1], 'penalty': [None, 'l1', 'l2']}
    mod, mods = train("PRUEBA LOGISTIC REGRESSION", dataframe, clasifier, grid, rmv_hg_corr=False)
    print(mod)
    print(mods)
