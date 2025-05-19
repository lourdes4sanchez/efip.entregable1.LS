# --- IMPORTACIÓN DE LIBRERÍAS ---
import pandas as pd # Manipulación de datos
import matplotlib.pyplot as plt  # Visualización estática
import seaborn as sns  # Visualización estadística
import numpy as np  # Cálculos numéricos

# --- SECCIÓN 1: CARGA Y COMPRENSIÓN DEL DATASET ---

# Cargar archivo CSV generado incorrectamente desde Excel (estructura irregular)
file_path = "casoefip.csv"
df_raw = pd.read_csv(file_path, header=None)

# Separar las columnas manualmente utilizando coma como delimitador
# ya que el archivo presenta todos los datos en una única columna
df_split = df_raw[0].str.split(",", expand=True)

# Usar la primera fila como encabezados y eliminarla del cuerpo de datos
df_split.columns = df_split.iloc[0]
df_split = df_split.drop(index=0).reset_index(drop=True)

# Eliminar columnas vacías o generadas incorrectamente por el formato del archivo
df_split = df_split.loc[:, ~df_split.columns.str.contains('^None|^Unnamed', na=False)]

# Seleccionar únicamente las columnas relevantes para el análisis
columnas_utiles = [
    'country', 'description', 'designation', 'points', 'price', 'province',
    'region_1', 'region_2', 'taster_name', 'taster_twitter_handle',
    'title', 'variety', 'winery'
]
df = df_split[columnas_utiles].copy()

# Convertir columnas 'points' y 'price' a formato numérico, ignorando errores
# Esto es necesario ya que algunas celdas pueden contener texto o valores vacíos
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Mostrar información básica del DataFrame cargado
print("Dimensiones del dataset:", df.shape)
print("Columnas:", df.columns.tolist())
print("Primeras filas:")
print(df.head())

# --- SECCIÓN 2: EXPLORACIÓN DE VARIABLES ---

# Mostrar tipos de datos y cantidad de valores no nulos por columna
print("\nInformación del dataframe:")
df.info()

# Mostrar cantidad de valores únicos por columna para conocer su variedad
print("\nValores únicos por columna:")
print(df.nunique().sort_values(ascending=False))

# Mostrar cantidad de valores nulos por columna
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Ver los 10 países más representados en el dataset
print("\nDistribución de los 10 países más frecuentes:")
print(df['country'].value_counts().head(10))

# Ver las 10 variedades de vino más comunes (puede haber valores atípicos si la columna fue mal alineada)
print("\nDistribución de las 10 variedades de uva más frecuentes:")
print(df['variety'].value_counts().head(10))

# Ver los catadores (taster_name) más frecuentes
print("\nCríticos más frecuentes:")
print(df['taster_name'].value_counts().head(10))

# Mostrar una muestra aleatoria de 5 filas para validar estructura de datos
print("\nMuestra aleatoria de 5 filas:")
print(df.sample(5))

# --- SECCIÓN 3: ESTADÍSTICAS DESCRIPTIVAS ---

# Filtrar solo filas con puntuaciones válidas (entre 80 y 100) para evitar outliers o errores de carga
df_valid = df[df['points'].between(80, 100)]

# Describir estadísticamente las variables numéricas relevantes
print("\nResumen estadístico de 'points' y 'price':")
print(df_valid[['points', 'price']].describe())

# Mostrar distribución de las puntuaciones válidas
print("\nDistribución de puntuaciones (points):")
print(df_valid['points'].value_counts().sort_index())

# Mostrar los 10 precios más altos registrados
print("\nTop 10 precios más altos:")
print(df['price'].sort_values(ascending=False).head(10))

# Mostrar los 10 precios válidos más bajos, excluyendo valores nulos o cero
print("\nTop 10 precios más bajos (mayores a 0):")
print(df[df['price'] > 0]['price'].sort_values().head(10))

# Mostrar registros que tienen puntuaciones fuera del rango esperado
print("\nVerificación de puntuaciones fuera del rango 80-100:")
print(df[~df['points'].between(80, 100)][['title', 'points']].head(10))

# Confirmar si existen registros con precio igual o menor a cero
print("\nPrecios menores o iguales a cero:")
print(df[df['price'] <= 0])

# --- SECCIÓN 4: VISUALIZACIÓN DE DATOS (ANÁLISIS RESUMIDO) ---

# Histograma de puntos (solo entre 80 y 100)
sns.histplot(df_valid['points'], bins=20)
plt.title("Distribución de puntuaciones válidas (80-100)")
plt.xlabel("Puntos")
plt.ylabel("Frecuencia")
plt.show()

# Histograma de precios usando logaritmo natural para evitar distorsión visual por valores extremos
sns.histplot(np.log1p(df[df['price'] > 0]['price']), bins=30)
plt.title("Distribución logarítmica de precios (log1p)")
plt.xlabel("log(Precio + 1)")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot limitado a precios menores a 500 USD para visualizar correctamente la mediana y dispersión sin distorsión
df_price_limited = df[df['price'] < 500]
plt.figure()
df_price_limited.boxplot(column='price', vert=True)
plt.title("Boxplot de precio (sin valores extremos > 500 USD)")
plt.ylabel("Precio (USD)")
plt.show()

# Gráfico de dispersión entre puntuación y precio, filtrando precios mayores a 500 para evitar distorsión
df_scatter = df_valid[df_valid['price'] < 500]
sns.scatterplot(x='points', y='price', data=df_scatter)
plt.title("Relación entre puntuación y precio (precio < 500 USD)")
plt.xlabel("Puntos")
plt.ylabel("Precio (USD)")
plt.show()

# --- SECCIÓN 5: TRATAMIENTO DE DATOS FALTANTES ---

# Calcular cantidad y porcentaje de valores faltantes por columna
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
print("\nValores faltantes por columna:")
print(pd.DataFrame({'Cantidad': missing_values, 'Porcentaje': missing_percent}))

# Imputar valores faltantes en la columna 'price' con la mediana (más robusta frente a outliers)
df['price'] = df['price'].fillna(df['price'].median())
print(f"\nSe imputaron los valores nulos de 'price' con la mediana: {df['price'].median()}")

# Mostrar columnas categóricas que aún contienen valores faltantes
categoricas_con_nulos = ['region_2', 'taster_name', 'taster_twitter_handle']
print("\nNulos en columnas categóricas conservados para revisión posterior:")
print(df[categoricas_con_nulos].isnull().sum())

# --- SECCIÓN 6: ANÁLISIS DE VALORES ATÍPICOS ---

# Calcular IQR para identificar precios atípicos
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Filtrar registros considerados outliers en precio
outliers_price = df[(df['price'] < limite_inferior) | (df['price'] > limite_superior)]

# Mostrar cantidad de outliers y los más costosos
print("\nCantidad de outliers en 'price':", outliers_price.shape[0])
print(outliers_price[['title', 'price']].sort_values(by='price', ascending=False).head(10))

# --- SECCIÓN 7: RECONOCIMIENTO DE PATRONES Y RELACIONES ---

# Agrupar por país y calcular promedio de puntuaciones y precios
print("\nPromedios por país:")
print(df_valid.groupby('country')[['points', 'price']].mean().sort_values(by='points', ascending=False).head(10))

# Agrupar por variedad y calcular promedios
print("\nPromedios por variedad:")
print(df_valid.groupby('variety')[['points', 'price']].mean().sort_values(by='points', ascending=False).head(10))

# Visualizar los países con mayor puntuación promedio
plt.figure(figsize=(10, 5))
df_valid.groupby('country')['points'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 países por puntuación promedio")
plt.ylabel("Puntos promedio")
plt.xlabel("País")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizar las variedades con mayor precio promedio
plt.figure(figsize=(10, 5))
df_valid.groupby('variety')['price'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='green')
plt.title("Top 10 variedades por precio promedio")
plt.ylabel("Precio promedio (USD)")
plt.xlabel("Variedad")
plt.xticks(rotation=45)
plt.tight_layout()



