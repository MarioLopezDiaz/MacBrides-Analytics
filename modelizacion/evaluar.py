import pandas as pd
import matplotlib.pyplot as plt
import os
import mlflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
import descarga_drive
from modelizacion.preparacion import preparar_datos
import warnings
warnings.filterwarnings('ignore')


def get_tomorrow_rets(df):
    """
        Obtiene los rendimientos de las acciones para el día siguiente.

        Parámetros:
        - df: DataFrame que contiene los datos.

        Devuelve:
        - tmrw_rets: Rendimientos de las acciones para el día siguiente.
    """
    tmrw_rets = df['Rets']
    tmrw_rets.index = df['Date']
    return tmrw_rets.shift(-1)


def score_rendimiento(y, y_pred, **kwargs):
    """
        Calcula el rendimiento de la estrategia de inversión.

        Parámetros:
        - y: Etiquetas reales.
        - y_pred: Etiquetas predichas.
        - **kwargs: Argumentos adicionales, se espera 'tomorrow_rets' que son los rendimientos de las acciones para el
        día siguiente.

        Devuelve:
        - score: Rendimiento de la estrategia de inversión.
    """
    tmrw_rets = kwargs.get('tomorrow_rets', None).loc[y.index]
    rets = (tmrw_rets + 1).prod()
    strat_rets = ((y_pred * tmrw_rets) + 1).prod()

    if rets == strat_rets or strat_rets == 0:
        return -100
    return strat_rets


def show_plot(y_pred, tomorrow_rets, ax):
    """
        Muestra un gráfico comparando los rendimientos de las acciones con los de la estrategia de inversión.

        Parámetros:
        - y_pred: Etiquetas predichas por el modelo.
        - tomorrow_rets: Rendimientos de las acciones para el día siguiente.
        - ax: Eje en el que se mostrará el gráfico.

        Devuelve:
        - No devuelve ningún valor, pero muestra un gráfico en el eje dado.
    """
    # SE CALCULAN LAS GANACIAS REALES DE LA ACCIÓN Y LAS DE LA ESTRATEGIA IA
    strat = tomorrow_rets * y_pred
    strat_return = ((strat + 1).cumprod() - 1).iloc[-2]
    stock_return = ((tomorrow_rets + 1).cumprod() - 1).iloc[-2]
    print('STOCK RETURN:', stock_return)
    print('STRAT_RETURN:', strat_return)

    # SE MUESTRA UN GRÁFICO CON LA COMPARATIVA DE AMBAS GANACIAS
    ax.plot(tomorrow_rets.index, (tomorrow_rets + 1).cumprod() - 1, label='Stock returns')
    ax.plot(tomorrow_rets.index, (strat + 1).cumprod() - 1, label='Strat returns')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Returns')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()


def evaluar(clf, x_train, x_test, y_train, y_test, tomorrow_rets, train_size=.8, show_plots=False, ax_val=None,
            ax_test=None):
    from modelizacion.train import cv_selection
    """
        Evalúa el rendimiento del clasificador en un conjunto de datos de validación y en el test final.

        Parámetros:
        - clf: Clasificador ya entrenado.
        - x_train: Características del conjunto de entrenamiento.
        - x_test: Características del conjunto de prueba.
        - y_train: Etiquetas del conjunto de entrenamiento.
        - y_test: Etiquetas del conjunto de prueba.
        - tomorrow_rets: Rendimientos de las acciones para el día siguiente.
        - train_size: Proporción del conjunto de entrenamiento respecto al total (por defecto 0.8).
        - show_plots: Booleano que indica si se deben mostrar los gráficos de rendimiento (por defecto False).
        - ax_val: Eje donde se mostrará el gráfico de rendimiento de validación (opcional).
        - ax_test: Eje donde se mostrará el gráfico de rendimiento de test (opcional).

        Devuelve:
        - (av, bav, rev, pv, rsv): Métricas de rendimiento en el conjunto de validación.
        - (at, bat, ret, pt, rst): Métricas de rendimiento en el conjunto de test.
    """
    train_indices, validation_indices = cv_selection(len(x_train), train_size)[0]
    x_t_train, y_t_train = x_train.iloc[train_indices], y_train.iloc[train_indices]
    x_validation, y_validation = x_train.iloc[validation_indices], y_train.iloc[validation_indices]
    clf.fit(x_t_train, y_t_train)
    y_pred_val = clf.predict(x_validation)
    a_v = round(accuracy_score(y_validation, y_pred_val), 5)
    ba_v = round(balanced_accuracy_score(y_validation, y_pred_val), 5)
    re_v = round(recall_score(y_validation, y_pred_val), 5)
    p_v = round(precision_score(y_validation, y_pred_val), 5)
    rs_v = score_rendimiento(y_validation, y_pred_val, tomorrow_rets=tomorrow_rets)

    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    a_t = round(accuracy_score(y_test, y_pred_test), 5)
    ba_t = round(balanced_accuracy_score(y_test, y_pred_test), 5)
    re_t = round(recall_score(y_test, y_pred_test), 5)
    p_t = round(precision_score(y_test, y_pred_test), 5)
    rs_t = score_rendimiento(y_test, y_pred_test, tomorrow_rets=tomorrow_rets)
    print('Matriz de confusión en test:\n', confusion_matrix(y_test, y_pred_test))

    if show_plots:
        show_plot(y_pred_val, tomorrow_rets.loc[y_validation.index], ax_val)
        show_plot(y_pred_test, tomorrow_rets.loc[y_test.index], ax_test)

    return (a_v, ba_v, re_v, p_v, rs_v), (a_t, ba_t, re_t, p_t, rs_t)


def get_clf(archivo, row):
    """
        Crea un clasificador basado en el archivo y la fila dada. La fila se obtiene de las bases de datos de MLFlow.

        Parámetros:
        - archivo: Nombre del archivo del modelo.
        - row: Fila de parámetros del modelo.

        Devuelve:
        - clf: Clasificador creado según los parámetros dados.
    """
    if 'LR' in archivo:
        return LogisticRegression(C=float(row['C']),
                                  penalty=(eval(row['penalty']) if row['penalty'] == 'None' else row['penalty']),
                                  solver=row['solver'], class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    elif 'RF' in archivo:
        return RandomForestClassifier(n_estimators=int(row['n_estimators']), criterion=row['criterion'],
                                      max_depth=int(row['max_depth']), min_samples_leaf=int(row['min_samples_leaf']),
                                      max_features=(row['max_features']
                                                    if row['max_features'] == 'sqrt'
                                                    else int(row['max_features'])),
                                      class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    elif 'MLP' in archivo:
        return MLPClassifier(hidden_layer_sizes=eval(row['hidden_layer_sizes']), batch_size=int(row['batch_size']),
                             activation=row['activation'], alpha=float(row['alpha']), max_iter=int(row['max_iter']),
                             early_stopping=True, random_state=RANDOM_STATE)


def analizar_experimentos(symbol, path, num_exp_model=5, show_plots=True):
    """
        Analiza los experimentos realizados sobre los datos del símbolo dado y muestra las métricas de evaluación.

        Parámetros:
        - symbol: Símbolo del activo financiero sobre el que se realizan los experimentos.
        - path: Ruta donde se encuentran los registros de los experimentos.
        - num_exp_model: Número de modelos experimentales a considerar por cada experimento.
        - show_plots: Booleano que indica si se deben mostrar los gráficos de rendimiento.

        Devuelve:
        - None
    """
    data = pd.read_parquet(f'../data/final_{symbol}.parquet').rename(columns={'Adj Close': 'Adj_Close'})
    if show_plots:
        fig_val, axes_val = plt.subplots(nrows=3, ncols=num_exp_model, figsize=(20, 10), sharey='all', sharex='all')
        fig_test, axes_test = plt.subplots(nrows=3, ncols=num_exp_model, figsize=(20, 10), sharey='all', sharex='all')
        fig_val.suptitle(f'VAL_{symbol}')
        fig_test.suptitle(f'TEST_{symbol}')
        i, j = 0, 0
    tomorrow_rets = get_tomorrow_rets(data)
    print(symbol)
    for archivo in os.listdir(path):
        path_db = f'sqlite:///{path}/{archivo}'
        mlflow.set_tracking_uri(path_db)
        exps = mlflow.search_experiments()
        for e in [e for e in exps if f'_{symbol}' in e.name and e.name != 'Default']:
            print('##### ', e.name)
            runs = mlflow.search_runs(e.experiment_id)
            columns_metrics, columns_params, run_name = ([c for c in runs.columns if 'metrics' in c],
                                                         [c for c in runs.columns if 'params' in c],
                                                         [c for c in runs.columns if 'runName' in c])
            runs = runs[columns_metrics + columns_params + run_name]
            runs.columns = [c.replace("metrics.", "").replace("params.", "").replace("tags.mlflow.", "")
                            for c in runs.columns]
            runs.sort_values(by='Rendimiento_stock_test', ascending=False, inplace=True)
            for k, row in runs.iloc[:num_exp_model].iterrows():
                cols = (row['Cols']
                        .strip("[]")
                        .replace("'", "")
                        .replace("\n", "")
                        .replace("Adj Close", "Adj_Close")
                        .split())
                print('################', k)
                data_aux = data.copy()
                x_train, x_test, y_train, y_test = preparar_datos(data_aux,
                                                                  eval(row['Remove_corr']),
                                                                  eval(row['Add_feat_sqr']),
                                                                  eval(row['Trans_data']),
                                                                  eval(row['Scaler'] if row['Trans_data'] else 'None'),
                                                                  row['Cols_to_scale'])
                x_train, x_test = x_train.loc[:, cols], x_test.loc[:, cols]
                clf = get_clf(archivo, row)
                title = f"{archivo}_{row['runName']}"
                print(title)
                data_val, (at, bat, ret, pt, rst) = evaluar(clf, x_train, x_test, y_train, y_test, tomorrow_rets,
                                                            .8, show_plots,
                                                            None if not show_plots else axes_val[i, j],
                                                            None if not show_plots else axes_test[i, j])
                print('Accuracy test:', at)
                print('Balanced accuracy test:', bat)
                print('Recall test:', ret)
                print('Precision test:', pt)
                print('Rendimiento test:', rst)
                if show_plots:
                    axes_val[i, j].set_title(title, fontsize=6)
                    axes_test[i, j].set_title(title, fontsize=6)
                    axes_val[i, j].tick_params(axis='x', rotation=45)
                    axes_test[i, j].tick_params(axis='x', rotation=45)
                    j += 1
                    if j == num_exp_model:
                        j = 0
                        i += 1
    if show_plots:
        fig_val.tight_layout()
        fig_test.tight_layout()
        if not os.path.exists('../resultados/rets_bestTest'):
            os.makedirs('../resultados/rets_bestTest')
        fig_val.savefig(f'../resultados/rets_bestTest/{symbol}_val.png')
        fig_test.savefig(f'../resultados/rets_bestTest/{symbol}_test.png')


if __name__ == '__main__':
    descarga_drive.main('archivos_final.txt')
    RANDOM_STATE = 42
    symbols = ['AAPL', 'F', 'GOOG', 'META', 'MSFT', 'TSLA']
    path_dbs = "dbs"
    for s in symbols:
        analizar_experimentos(s, path_dbs, 3, True)
