# =============================================================================
# 1. Importação de Bibliotecas
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, 
                            r2_score, 
                            mean_absolute_error)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# =============================================================================
# 2. Carregamento e Exploração Inicial dos Dados
# =============================================================================
def carregar_dados(caminho):
    """Carrega os dados do arquivo CSV"""
    dados = pd.read_csv(caminho, sep=';', decimal=',')
    return dados

# Carrega os dados (ajuste o caminho do arquivo conforme necessário)
try:
    dados = carregar_dados('dados.csv')
except FileNotFoundError:
    exit()
def carregar_dados2():
    """Carrega os dados da tabela MySQL"""
    try:
        # Consulta todos os registros da tabela
        registros = Dados.query.all()

        # Converte para DataFrame
        dados_dict = [registro.to_dict() for registro in registros]
        df = pd.DataFrame(dados_dict)

        # Remove a coluna 'id' se não for necessária
        df.drop('id', axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        print(f"Erro ao carregar dados do MySQL: {str(e)}")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro

# =============================================================================
# 4. Preparação dos Dados
# =============================================================================
def preparar_dados(df, target_col, test_size=0.2, random_state=42):
    """Prepara os dados para modelagem"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Preparação dos dados
X_train, X_test, y_train, y_test = preparar_dados(dados, 'Target')

# =============================================================================
# 5. Modelagem e Avaliação
# =============================================================================
def treinar_modelo(X_train, y_train):
    """Treina o modelo de regressão linear"""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo com diversas métricas"""
    y_pred = modelo.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, rmse, mae, r2

# Treinamento e avaliação
modelo = treinar_modelo(X_train, y_train)
y_pred, mse, rmse, mae, r2 = avaliar_modelo(modelo, X_test, y_test)

# Validação cruzada
cv_scores = cross_val_score(modelo, dados.drop(columns=['Target']), dados['Target'], cv=5, scoring='r2')

# =============================================================================
# 6. Análise dos Resultados
# =============================================================================
def analisar_coeficientes(modelo, feature_names):
    """Analisa e exibe os coeficientes do modelo"""
    coeficientes = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': modelo.coef_,
        'Magnitude Absoluta': np.abs(modelo.coef_)
    }).sort_values('Magnitude Absoluta', ascending=False)
    
    return coeficientes

def plot_importancia(coeficientes, save_path=None):
    """Gráfico de importância das variáveis"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coeficiente', y='Variável', 
                data=coeficientes.sort_values('Coeficiente', ascending=False))
    plt.title('Importância das Variáveis no Modelo')
    plt.xlabel('Coeficiente')
    plt.ylabel('Variável')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_reais_vs_previstos(y_test, y_pred, save_path=None):
    """Gráfico de valores reais vs previstos"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Valores Reais vs Valores Previstos')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_residuos(y_test, y_pred, save_path=None):
    """Gráfico de análise de resíduos"""
    residuos = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True)
    plt.title('Distribuição dos Resíduos')
    plt.xlabel('Resíduos')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

# Execução das análises
coeficientes = analisar_coeficientes(modelo, X_train.columns)
plot_importancia(coeficientes, 'importancia_variaveis.png')
plot_reais_vs_previstos(y_test, y_pred, 'reais_vs_previstos.png')
plot_residuos(y_test, y_pred, 'distribuicao_residuos.png')

# =============================================================================
# 7. Análise de Multicolinearidade (VIF)
# =============================================================================
def calcular_vif(X):
    """Calcula o Fator de Inflação de Variância"""
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(len(X.columns))]
    return vif_data.sort_values('VIF', ascending=False)

# =============================================================================
# 8. Modelo Estatístico com Statsmodels (para análise detalhada)
# =============================================================================
def modelo_statsmodels(X, y):
    """Executa modelo estatístico com análise detalhada"""
    X_sm = sm.add_constant(X)
    modelo_sm = sm.OLS(y, X_sm).fit()
    return modelo_sm


modelo_sm = modelo_statsmodels(dados.drop(columns=['Target']), dados['Target'])


# =============================================================================
# 9. Salvando Resultados
# =============================================================================
def salvar_resultados(modelo, coeficientes, metricas, filename='resultados.txt'):
    """Salva os principais resultados em um arquivo de texto"""
    with open(filename, 'w') as f:
        f.write("RESULTADOS DA ANÁLISE DE REGRESSÃO\n")
        f.write("="*50 + "\n\n")
        
        f.write("Métricas do Modelo:\n")
        f.write(f"- R²: {metricas['r2']:.4f}\n")
        f.write(f"- RMSE: {metricas['rmse']:.4f}\n")
        f.write(f"- MAE: {metricas['mae']:.4f}\n\n")
        
        f.write("Coeficientes do Modelo:\n")
        f.write(coeficientes.to_string())
        f.write("\n\n")
        
        f.write("Modelo Estatístico:\n")
        f.write(str(modelo_sm.summary()))

# Dicionário com as métricas
metricas = {
    'r2': r2,
    'rmse': rmse,
    'mae': mae
}

salvar_resultados(modelo, coeficientes, metricas)