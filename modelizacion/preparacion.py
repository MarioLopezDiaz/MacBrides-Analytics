import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def df_to_numeric(df):
    """
        Convierte columnas categóricas en valores numéricos y reformatea la columna de fecha poniendola como índice

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - df_transform: DataFrame con columnas convertidas a valores numéricos y la columna de fecha reformateada.
    """
    df_transform = df.copy()
    df_transform['Sent'] = np.where(df_transform['Sent'] == 'neutral',
                                    0,
                                    np.where(df_transform['Sent'] == 'positive', 1, -1))
    df_transform['Date'] = pd.to_datetime(df_transform['Date'])
    df_transform.index = df_transform['Date']
    df_transform.drop('Date', axis=1, inplace=True)
    return df_transform


def variables_corr_count(df, cota_corr=.85, method='spearman'):
    """
        Calcula las parejas de variables con alta correlación en un DataFrame.

        Parámetros:
        - df: DataFrame que contiene los datos.
        - cota_corr: Umbral de correlación a partir del cual se considera alta correlación (por defecto 0.85).
        - method: Método utilizado para calcular la correlación (por defecto 'spearman').

        Devuelve:
        - high_correlation_pairs: Lista de pares de variables con alta correlación.
    """
    correlation_matrix = df.corr(method=method)
    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if correlation_matrix.columns[i] != correlation_matrix.columns[j]:
                if abs(correlation_matrix.iloc[i, j]) > cota_corr:
                    pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
                    high_correlation_pairs.append(pair)
    return high_correlation_pairs


def remove_high_corr(df, cota_corr=.85):
    """
        Elimina las variables con alta correlación en un DataFrame.

        Parámetros:
        - df: DataFrame que contiene los datos.
        - cota_corr: Umbral de correlación a partir del cual se considera alta correlación (por defecto 0.85).

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df eliminando las variables
        con alta correlación.
    """
    for cota in range(100, int(cota_corr*100)-1, -2):
        while True:
            high_correlation_pairs = variables_corr_count(df, cota/100)
            if not high_correlation_pairs:
                break
            columna_borrar = pd.DataFrame(high_correlation_pairs)[0].value_counts().idxmax()
            df.drop(columna_borrar, axis=1, inplace=True)


def add_features_squared(df):
    """
        Agrega las características al cuadrado al DataFrame.

        Parámetros:
        - df: DataFrame que contiene los datos.

        Devuelve:
        - df_aux: DataFrame con características al cuadrado agregadas.
    """
    df_aux = df.copy()
    new_columns = {}
    for c in df_aux.columns:
        if c != 'Invest' and df_aux[c].nunique() > 2:
            new_columns[c+str('^2')] = pow(df_aux[c], 2)
    new_df = pd.DataFrame(new_columns)
    df_aux = pd.concat([df_aux, new_df], axis=1)
    return df_aux


def get_columns_to_scale(df, columns_to_scale='continuous'):
    """
        Obtiene las columnas que se deben escalar.

        Parámetros:
        - df: DataFrame que contiene los datos.
        - columns_to_scale: Tipo de columnas a escalar ('continuous' para las columnas con más de 3 valores únicos,
          'all' para todas las columnas, 'big' para las columnas de gran magnitud, None para ninguna columna).

        Devuelve:
        - Lista de columnas que se deben escalar.
    """
    if columns_to_scale == 'continuous':
        return [c for c in df if df[c].nunique() > 2]
    elif columns_to_scale == 'all':
        return df.columns
    elif columns_to_scale == 'big':
        return ['Volume', 'volume_obv', 'volume_adi', 'volume_fi', 'volume_vpt', 'volume_nvi', 'others_cr']
    elif columns_to_scale is None:
        return columns_to_scale
    raise


def split_data(df, target='Invest', test_size=.2):
    """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Parámetros:
        - df: DataFrame que contiene los datos.
        - target: Nombre de la variable objetivo (por defecto 'Invest').
        - test_size: Proporción del conjunto de datos que se utilizará como conjunto de prueba (por defecto 0.2).

        Devuelve:
        - x_train: Datos de entrenamiento.
        - x_test: Datos de prueba.
        - y_train: Etiquetas de entrenamiento.
        - y_test: Etiquetas de prueba.
    """
    # SEPARAMOS VARIABLES PREDICTORAS Y VARIABLES RESPUESTA
    x = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(x, y, test_size=test_size, shuffle=False)


def transform_data(df, cols_scale, scaler=StandardScaler()):
    """
        Transforma los datos, escalando las columnas especificadas y dividiendo los datos en
        conjuntos de entrenamiento y prueba. Escala los datos (tanto train como test) a partir de la
        distribución de las variables de x_train

        Parámetros:
        - df: DataFrame que contiene los datos.
        - cols_scale: Lista de columnas a escalar.
        - scaler: Objeto de escalado (por defecto StandardScaler()).

        Devuelve:
        - x_train_scaled: Datos de entrenamiento escalados.
        - x_test_scaled: Datos de prueba escalados.
        - y_train: Etiquetas de entrenamiento.
        - y_test: Etiquetas de prueba.
    """
    cols_scale = [c for c in cols_scale if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_scale]
    other_cols.remove('Invest')
    x_train, x_test, y_train, y_test = split_data(df)
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train[cols_scale]), columns=cols_scale, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test[cols_scale]), columns=cols_scale, index=x_test.index)
    return (pd.concat([x_train_scaled, x_train[other_cols]], axis=1),
            pd.concat([x_test_scaled, x_test[other_cols]], axis=1), y_train, y_test)


def preparar_datos(df, rmv_hg_corr=True, add_feat_sq=True, trans_data=True, scaler=StandardScaler(),
                   columns_to_scale='continuous'):
    """
        Prepara los datos para el modelado, realizando una serie de transformaciones.

        Parámetros:
        - df: DataFrame que contiene los datos.
        - rmv_hg_corr: Booleano que indica si se deben eliminar variables con alta correlación (por defecto True).
        - add_feat_sq: Booleano que indica si se deben agregar características al cuadrado (por defecto True).
        - trans_data: Booleano que indica si se deben transformar los datos (por defecto True).
        - scaler: Objeto de escalado (por defecto StandardScaler()).
        - columns_to_scale: Tipo de columnas a escalar ('continuous' para las columnas con más de 3 valores únicos,
          'all' para todas las columnas, 'big' para las columnas de gran magnitud, None para ninguna columna)

        Devuelve:
        - Datos preparados para modelado (dependiendo de la configuración, puede devolver diferentes conjuntos de datos)
    """
    aux = df_to_numeric(df)
    if rmv_hg_corr:
        remove_high_corr(aux)
    if add_feat_sq:
        aux = add_features_squared(aux)
    if trans_data:
        cols_scale = get_columns_to_scale(df, columns_to_scale)
        return transform_data(aux, cols_scale, scaler)
    return split_data(aux)


if __name__ == '__main__':
    dataframe = pd.read_parquet('../data/final_AAPL.parquet')
    print(dataframe)
    print(preparar_datos(dataframe))
    print(preparar_datos(dataframe, rmv_hg_corr=False))
    print(preparar_datos(dataframe, add_feat_sq=False))
    print(preparar_datos(dataframe, rmv_hg_corr=False, add_feat_sq=False))
    print(preparar_datos(dataframe, columns_to_scale='big'))
