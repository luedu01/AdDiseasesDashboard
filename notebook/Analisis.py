import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# Configuración inicial
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# 1. Carga y exploración inicial de datos
def load_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    
    print("=== Dimensiones del dataset ===")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    
    print("\n=== Primeras filas ===")
    display(df.head())
    
    print("\n=== Resumen estadístico ===")
    display(df.describe(include='all'))
    
    print("\n=== Tipos de datos y valores nulos ===")
    display(df.info())
    
    return df

# 2. Análisis de correlación
def correlation_analysis(df, target_col='target'):
    # Codificar target (healthy=0, diseased=1)
    df[target_col] = df[target_col].map({'healthy': 0, 'diseased': 1})
    
    # Seleccionar solo variables numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_df = df[numeric_cols]
    
    # Matriz de correlación
    corr_matrix = numeric_df.corr()
    
    # Correlación con la variable objetivo
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    print("\n=== Correlación con la variable objetivo ===")
    display(target_corr)
    
    # Visualización
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('Matriz de Correlación')
    plt.show()
    
    return target_corr

# 3. Pruebas de significancia para variables categóricas
def categorical_analysis(df, target_col='target'):
    # Seleccionar variables categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in cat_cols if col != target_col]
    
    # Codificar variables categóricas para chi-cuadrado
    le = LabelEncoder()
    df_encoded = df.copy()
    
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    # Prueba chi-cuadrado
    X = df_encoded[cat_cols]
    y = df_encoded[target_col]
    
    chi2_results = []
    p_values = []
    
    for feature in X.columns:
        chi2_stat, p_val = stats.chisquare(pd.crosstab(X[feature], y).values)
        chi2_results.append(chi2_stat)
        p_values.append(p_val)
    
    chi2_df = pd.DataFrame({
        'Variable': cat_cols,
        'Chi2_Statistic': chi2_results,
        'P_Value': p_values
    }).sort_values('P_Value')
    
    print("\n=== Significancia estadística (chi-cuadrado) ===")
    display(chi2_df)
    
    return chi2_df

# 4. Análisis de importancia con Random Forest
def feature_importance_analysis(df, target_col='target'):
    # Preparar datos
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Codificar variables categóricas
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Entrenar modelo preliminar
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_encoded, y)
    
    # Importancia de características
    importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n=== Importancia de características (Random Forest) ===")
    display(importance.head(15))
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(15))
    plt.title('Top 15 Variables Más Importantes')
    plt.show()
    
    return importance

# 5. Análisis de multicolinealidad
def multicollinearity_analysis(df, target_col='target'):
    # Seleccionar solo variables numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    numeric_df = df[numeric_cols]
    
    # Calcular VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(len(numeric_cols))]
    
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("\n=== Factor de Inflación de Varianza (VIF) ===")
    print("VIF > 10 indica alta multicolinealidad")
    display(vif_data)
    
    # Identificar pares altamente correlacionados
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(column, upper[column].idxmax(), upper[column].max()) 
                       for column in upper.columns if upper[column].max() > 0.8]
    
    print("\n=== Pares de variables altamente correlacionadas (>0.8) ===")
    for pair in high_corr_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")
    
    return vif_data, high_corr_pairs

# 6. Función principal que ejecuta todo el análisis
def full_analysis(file_path):
    # 1. Carga y exploración inicial
    print("\n" + "="*50)
    print("1. CARGA Y EXPLORACIÓN INICIAL DE DATOS")
    print("="*50)
    df = load_and_explore_data(file_path)
    
    # 2. Análisis de correlación
    print("\n" + "="*50)
    print("2. ANÁLISIS DE CORRELACIÓN")
    print("="*50)
    target_corr = correlation_analysis(df)
    
    # 3. Análisis de variables categóricas
    print("\n" + "="*50)
    print("3. ANÁLISIS DE VARIABLES CATEGÓRICAS")
    print("="*50)
    chi2_df = categorical_analysis(df)
    
    # 4. Importancia de características
    print("\n" + "="*50)
    print("4. IMPORTANCIA DE CARACTERÍSTICAS CON RANDOM FOREST")
    print("="*50)
    importance_df = feature_importance_analysis(df)
    
    # 5. Análisis de multicolinealidad
    print("\n" + "="*50)
    print("5. ANÁLISIS DE MULTICOLINEALIDAD")
    print("="*50)
    vif_data, high_corr_pairs = multicollinearity_analysis(df)
    
    return {
        'data': df,
        'target_correlation': target_corr,
        'chi2_results': chi2_df,
        'feature_importance': importance_df,
        'vif_data': vif_data,
        'high_corr_pairs': high_corr_pairs
    }

# Ejecutar análisis completo
if __name__ == "__main__":
    file_path = "health_lifestyle_classification.csv"  # Cambiar por la ruta correcta
    analysis_results = full_analysis(file_path)