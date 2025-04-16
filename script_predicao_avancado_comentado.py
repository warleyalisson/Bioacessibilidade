
# -*- coding: utf-8 -*-
"""
SCRIPT AVANÇADO DE MACHINE LEARNING PARA PREDIÇÃO DE PROPRIEDADES FUNCIONAIS EM ALIMENTOS

Este script realiza:
1. Carregamento de dados de um arquivo CSV
2. Pré-processamento (tratamento de NaNs, codificação, escalonamento)
3. Treinamento de modelo Random Forest com pipeline
4. Avaliação com RMSE, R² e validação cruzada
5. Visualização dos resultados e importâncias das variáveis

Requisitos: pip install pandas scikit-learn matplotlib seaborn numpy openpyxl

Autor: [Seu Nome]
Data: [Atual]
"""

# Importações
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configurações gerais
FILE_PATH = 'dados_amostras.csv'
TARGET_COLUMN = 'bioacessibilidade'
MISSING_THRESHOLD = 0.8
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100

# Função: carregamento dos dados
def load_data(filepath: str) -> Optional[pd.DataFrame]:
    try:
        print(f"Carregando dados de {filepath}")
        df = pd.read_csv(filepath)
        print(f"{df.shape[0]} linhas e {df.shape[1]} colunas carregadas.")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Função: pré-processamento
def preprocess_data(df: pd.DataFrame, target_col: str, missing_thresh: float) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str]]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.dropna(axis=1, thresh=int(len(df) * missing_thresh))

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    feature_names_out = preprocessor.fit(X).get_feature_names_out()
    return X, y, preprocessor, list(feature_names_out)

# Função: treinamento e avaliação
def train_evaluate_model(X, y, preprocessor, feature_names_out, test_size, cv_folds, n_estimators, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    cv_rmse = -cross_val_score(pipeline, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)

    importances = pipeline.named_steps['regressor'].feature_importances_
    importances_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    results_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})

    return pipeline, {'RMSE': rmse, 'R2': r2}, pd.Series(cv_rmse), importances_df, results_df

# Função: visualização de resultados
def plot_results(results_df, importances_df, target_col, metrics):
    sns.set(style='whitegrid')

    # Real vs Previsto
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Real', y='Previsto', data=results_df)
    plt.plot([results_df.min().min(), results_df.max().max()], [results_df.min().min(), results_df.max().max()], 'r--')
    plt.title(f'Real vs Previsto - {target_col}')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Previsto')
    plt.text(0.05, 0.95, f"RMSE: {metrics['RMSE']:.3f}
R²: {metrics['R2']:.3f}", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(f'plot_real_vs_previsto_{target_col}.png')
    plt.show()

    # Importância das features
    top_n = min(20, len(importances_df))
    plt.figure(figsize=(10, top_n * 0.4))
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(top_n))
    plt.title(f'Importância das Variáveis - {target_col}')
    plt.tight_layout()
    plt.savefig(f'plot_importancia_features_{target_col}.png')
    plt.show()

# Execução principal
def main():
    df = load_data(FILE_PATH)
    if df is None: return
    X, y, preprocessor, features = preprocess_data(df, TARGET_COLUMN, MISSING_THRESHOLD)
    model, metrics, cv_scores, importances_df, results_df = train_evaluate_model(X, y, preprocessor, features, TEST_SIZE, CV_FOLDS, N_ESTIMATORS, RANDOM_STATE)
    plot_results(results_df, importances_df, TARGET_COLUMN, metrics)
    print("
Resumo:")
    print(f"RMSE: {metrics['RMSE']:.3f}")
    print(f"R²: {metrics['R2']:.3f}")
    print(f"CV RMSE Médio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

if __name__ == "__main__":
    main()
