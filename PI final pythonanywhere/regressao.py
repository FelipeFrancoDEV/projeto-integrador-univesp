# regressao.py (Versão Melhorada - Correção Sintaxe L153 v4)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Backend não interativo para Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")


def calcular_vif(X):
    """Calcula o Fator de Inflação de Variância (VIF) para cada variável.

    Args:
        X (pd.DataFrame): DataFrame com as variáveis independentes.

    Returns:
        pd.DataFrame: DataFrame com as variáveis e seus respectivos VIFs,
                      ordenado de forma decrescente pelo VIF.
    """
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    # Garante que os valores são numéricos e finitos para VIF
    X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Adiciona constante para cálculo de VIF robusto
    X_numeric_const = sm.add_constant(X_numeric, prepend=False)
    try:
        vif_data["VIF"] = [variance_inflation_factor(X_numeric_const.values, i)
                           for i in range(X_numeric.shape[1])] # Calcula VIF para features originais
    except Exception as e:
        vif_data["VIF"] = np.nan
    return vif_data.sort_values("VIF", ascending=False)

def analisar_variancia_pca(pca):
    """Analisa e formata a variância explicada pelos componentes PCA.

    Args:
        pca (PCA): Objeto PCA ajustado.

    Returns:
        pd.DataFrame: DataFrame com a variância explicada por componente.
    """
    variancia_df = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(pca.n_components_)],
        "Variância Individual": pca.explained_variance_ratio_,
        "Variância Acumulada": np.cumsum(pca.explained_variance_ratio_)
    })
    return variancia_df

def diagnosticar_dados(df, target_col):
    """Realiza um diagnóstico inicial dos dados, incluindo cálculo de VIF.

    Args:
        df (pd.DataFrame): DataFrame completo.
        target_col (str): Nome da coluna alvo.

    Returns:
        dict: Dicionário contendo o DataFrame de VIF e os nomes das features.
              Retorna None se a coluna alvo não for encontrada.
    """
    if target_col not in df.columns:
        return None

    X = df.drop(columns=[target_col])
    vif_original = calcular_vif(X)

    return {
        "vif_original": vif_original,
        "feature_names_original": X.columns.tolist()
    }

def preprocessar_dados(df, target_col, aplicar_pca=False, padronizar=True,
                       n_components=0.95, test_size=0.2, random_state=42):
    """Prepara e divide os dados para modelagem, com opções de pré-processamento.

    Args:
        df (pd.DataFrame): DataFrame completo.
        target_col (str): Nome da coluna alvo.
        aplicar_pca (bool, optional): Se True, aplica PCA. Defaults to False.
        padronizar (bool, optional): Se True, aplica StandardScaler.
                                     Necessário se aplicar_pca=True. Defaults to True.
        n_components (float or int, optional): Número de componentes para PCA.
                                                Se float (0 < n < 1), é a variância explicada.
                                                Se int, é o número de componentes.
                                                Defaults to 0.95.
        test_size (float, optional): Proporção do dataset para o conjunto de teste.
                                     Defaults to 0.2.
        random_state (int, optional): Seed para reprodutibilidade na divisão.
                                      Defaults to 42.

    Returns:
        dict: Dicionário contendo os conjuntos de treino/teste (X_train, X_test,
              y_train, y_test) e opcionalmente os objetos scaler e pca ajustados,
              e informações sobre a variância do PCA.
              Retorna None se a coluna alvo não for encontrada.
    """
    if target_col not in df.columns:
        return None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_processado = X.copy()
    scaler = None
    pca = None
    variancia_pca_df = None
    feature_names = X.columns.tolist()

    if aplicar_pca and not padronizar:
        warnings.warn("PCA geralmente requer padronização. Ativando padronização.")
        padronizar = True

    if padronizar:
        scaler = StandardScaler()
        X_processado_scaled = scaler.fit_transform(X_processado)
        # Manter como DataFrame para consistência se não aplicar PCA
        if not aplicar_pca:
             X_processado = pd.DataFrame(X_processado_scaled, columns=X.columns, index=X.index)
        else:
             X_processado = X_processado_scaled # PCA opera em array numpy

    if aplicar_pca:
        pca = PCA(n_components=n_components)
        # Se padronizou, X_processado é um array numpy
        # Se não padronizou (caso raro), precisa converter para array
        if not isinstance(X_processado, np.ndarray):
             X_processado_array = X_processado.values
        else:
             X_processado_array = X_processado
        X_processado = pca.fit_transform(X_processado_array)
        variancia_pca_df = analisar_variancia_pca(pca)
        feature_names = [f"PC{i+1}" for i in range(pca.n_components_)]
        # Correção de sintaxe na linha abaixo (usando variável intermediária):
        variancia_acumulada_final = variancia_pca_df["Variância Acumulada"].iloc[-1]


    X_train, X_test, y_train, y_test = train_test_split(
        X_processado, y, test_size=test_size, random_state=random_state
    )

    resultado = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names # Nomes das features usadas no treino (originais ou PCs)
    }

    # Adicionar objetos de pré-processamento se foram usados
    if scaler:
        resultado["scaler"] = scaler
    if pca:
        resultado["pca"] = pca
        resultado["variancia_pca"] = variancia_pca_df

    return resultado

# --- Funções de Modelagem --- #

def treinar_modelo_elasticnet(X_train, y_train, cv=5, random_state=42):
    """Treina modelo ElasticNet com otimização de hiperparâmetros.

    Args:
        X_train: Dados de treino (features).
        y_train: Variável alvo de treino.
        cv (int): Número de folds para validação cruzada.
        random_state (int): Seed para reprodutibilidade.

    Returns:
        ElasticNet: Modelo ElasticNet treinado e otimizado.
    """
    # Grade de parâmetros para ElasticNet
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Busca em grade com validação cruzada
    grid = GridSearchCV(
        ElasticNet(max_iter=10000, random_state=random_state),
        param_grid,
        cv=cv,
        n_jobs=1,  # Desativa processamento paralelo
        verbose=1,
        scoring='r2'
    )

    try:
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    except Exception as e:
        print(f"Erro durante treinamento: {str(e)}")
        raise



    # Criar modelo final com os melhores hiperparâmetros
    modelo_final = ElasticNet(**grid.best_params_, max_iter=10000, random_state=random_state)
    modelo_final.fit(X_train, y_train)

    return modelo_final

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo treinado no conjunto de teste.

    Args:
        modelo: Modelo treinado.
        X_test: Dados de teste (features).
        y_test: Variável alvo de teste.

    Returns:
        tuple: Contendo (y_pred, metricas), onde metricas é um dicionário
               com R², RMSE e MAE.
    """

    y_pred = modelo.predict(X_test)

    metricas = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    return y_pred, metricas

# --- Funções de Interpretação --- #

def calcular_importancia_features_pca(pca, modelo, feature_names_original):
    """Calcula importância das features originais quando PCA foi usado.

    Args:
        pca (PCA): Objeto PCA ajustado.
        modelo: Modelo treinado nos componentes PCA.
        feature_names_original (list): Lista dos nomes das variáveis originais.

    Returns:
        pd.DataFrame: DataFrame com a importância estimada das variáveis originais.
    """
    if not hasattr(modelo, "coef_"):

        return None
    if pca is None:

         return None

    try:
        # Importância dos componentes (abs(coef) do modelo)
        importancia_componentes = np.abs(modelo.coef_)

        # Contribuição das features para cada componente (abs)
        contrib_features = np.abs(pca.components_)

        # Importância agregada (produto escalar ponderado)
        importancia_features = contrib_features.T.dot(importancia_componentes)

        # Normalizar importância para soma 1 (opcional, para interpretação relativa)
        importancia_normalizada = importancia_features / importancia_features.sum()

        return pd.DataFrame({
            "Variável": feature_names_original,
            "Importância Estimada": importancia_normalizada # Usar a normalizada
        }).sort_values("Importância Estimada", ascending=False)

    except Exception as e:

        return None

def obter_coeficientes_modelo(modelo, feature_names):
    """Obtém os coeficientes do modelo treinado.

    Args:
        modelo: Modelo treinado.
        feature_names (list): Nomes das features usadas no modelo (originais ou PCs).

    Returns:
        pd.DataFrame: DataFrame com os coeficientes e suas magnitudes.
    """
    if not hasattr(modelo, "coef_"):
        return None

    coefs_df = pd.DataFrame({
        "Feature": feature_names,
        "Coeficiente": modelo.coef_,
        "Magnitude Absoluta": np.abs(modelo.coef_)
    }).sort_values("Magnitude Absoluta", ascending=False)
    return coefs_df

# --- Funções de Visualização (Refatoradas) --- #

def _setup_plot(figsize=(8, 6), title="", xlabel="", ylabel=""):
    """Configura uma nova figura e eixos Matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def _save_and_close_plot(fig, save_path):
    """Salva e fecha a figura Matplotlib."""
    if save_path:
        try:
            # Garante que o diretório existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Gráfico salvo em: {save_path}")
        except Exception as e:
            print(f"Erro ao salvar gráfico em {save_path}: {e}")
    plt.close(fig) # Fecha a figura para liberar memória

def plot_correlacao(df, save_path=None):
    """Gera e salva heatmap de correlação."""
    fig, ax = _setup_plot(figsize=(10, 8), title="Matriz de Correlação entre Variáveis")
    # Calcula a matriz de correlação apenas para colunas numéricas
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                linewidths=.5, cbar_kws={"shrink": .8}, mask=mask, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta layout para o título
    _save_and_close_plot(fig, save_path)

def plot_distribuicao(serie, var_name, save_path=None):
    """Gera e salva gráfico de distribuição (histograma + KDE)."""
    fig, ax = _setup_plot(figsize=(8, 5), title=f"Distribuição da Variável {var_name}",
                        xlabel="Valor", ylabel="Frequência")
    sns.histplot(serie, kde=True, ax=ax)
    fig.tight_layout()
    _save_and_close_plot(fig, save_path)

def plot_importancia(importancia_df, save_path=None):
    """Gera e salva gráfico de barras da importância das variáveis."""
    if importancia_df is None or importancia_df.empty:
        return
    fig, ax = _setup_plot(figsize=(8, 6), title="Importância Estimada das Variáveis Originais",
                        xlabel="Importância Relativa Estimada", ylabel="Variável")
    sns.barplot(x="Importância Estimada", y="Variável", data=importancia_df, ax=ax, palette="viridis")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_and_close_plot(fig, save_path)

def plot_reais_vs_previstos(y_test, y_pred, save_path=None):
    """Gera e salva gráfico de dispersão: valores reais vs. previstos."""
    fig, ax = _setup_plot(figsize=(7, 7), title="Valores Reais vs. Valores Previstos",
                        xlabel="Valores Reais", ylabel="Valores Previstos")
    min_val = min(y_test.min(), y_pred.min()) * 0.98
    max_val = max(y_test.max(), y_pred.max()) * 1.02
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y=x)")
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _save_and_close_plot(fig, save_path)

def plot_distribuicao_residuos(y_test, y_pred, save_path=None):
    """Gera e salva histograma da distribuição dos resíduos."""
    residuos = y_test - y_pred
    fig, ax = _setup_plot(figsize=(8, 5), title="Distribuição dos Resíduos (Erros do Modelo)",
                        xlabel="Resíduo (Real - Previsto)", ylabel="Frequência")
    sns.histplot(residuos, kde=True, ax=ax)
    ax.axvline(x=0, color="red", linestyle="--", label="Erro Zero")
    ax.legend()
    fig.tight_layout()
    _save_and_close_plot(fig, save_path)

def plot_qq_residuos(y_test, y_pred, save_path=None):
    """Gera e salva Q-Q plot dos resíduos."""
    residuos = y_test - y_pred
    fig, ax = _setup_plot(figsize=(6, 6), title="Q-Q Plot dos Resíduos vs. Normalidade")
    sm.qqplot(residuos, line="45", fit=True, ax=ax)
    ax.get_lines()[1].set_color("red") # Linha de referência vermelha
    ax.get_lines()[1].set_linestyle("--")
    ax.set_xlabel("Quantis Teóricos (Normal)")
    ax.set_ylabel("Quantis Amostrais (Resíduos)")
    fig.tight_layout()
    _save_and_close_plot(fig, save_path)

