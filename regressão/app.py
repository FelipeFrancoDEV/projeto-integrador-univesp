import os
from flask import Flask, render_template
from regressão import carregar_dados, preparar_dados, treinar_modelo, avaliar_modelo, analisar_coeficientes, calcular_vif, modelo_statsmodels
import matplotlib
matplotlib.use('Agg')  # Critical for Flask
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')  # Necessário para gerar gráficos em ambiente web

def plot_correlacao(df, save_path=None):
    """Gera e exibe matriz de correlação"""
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação entre Variáveis', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_distribuicao(serie, title, xlabel, save_path=None):
    """Gera gráfico de distribuição"""
    plt.figure(figsize=(8, 5))
    sns.histplot(serie, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequência')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residuos(y_test, y_pred, save_path=None):
    """Gráfico de análise de resíduos"""
    residuos = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True)
    plt.title('Distribuição dos Resíduos')
    plt.xlabel('Resíduos')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para executar a análise e mostrar resultados
@app.route('/analisar')
def analisar():
    try:
        # 1. Carregar e preparar dados
        dados = carregar_dados('dados.csv')
		#dados = carregar_dados()
		#if dados.empty:
            #return render_template('error.html', error="Nenhum dado encontrado no banco de dados")
        X_train, X_test, y_train, y_test = preparar_dados(dados, 'Target')
        
        # 2. Treinar e avaliar modelo
        modelo = treinar_modelo(X_train, y_train)
        y_pred, mse, rmse, mae, r2 = avaliar_modelo(modelo, X_test, y_test)
        
        # 3. Coletar informações para exibição
        coeficientes = analisar_coeficientes(modelo, X_train.columns)
        vif = calcular_vif(dados.drop(columns=['Target']))
        modelo_sm = modelo_statsmodels(dados.drop(columns=['Target']), dados['Target'])
        
        # 4. Garantir que a pasta de imagens existe
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 5. Gerar e salvar gráficos
        plot_paths = {}
        
        # Gráfico de Correlação
        plot_correlacao(dados, os.path.join(app.config['UPLOAD_FOLDER'], 'correlacao.png'))
        plot_paths['correlacao'] = 'images/correlacao.png'
        
        # Distribuição da Target
        plot_distribuicao(dados['Target'], 'Distribuição da Variável Target', 'Valor', 
                         os.path.join(app.config['UPLOAD_FOLDER'], 'distribuicao_target.png'))
        plot_paths['distribuicao'] = 'images/distribuicao_target.png'
        
        # Importância das Variáveis
        plot_importancia(coeficientes, os.path.join(app.config['UPLOAD_FOLDER'], 'importancia_variaveis.png'))
        plot_paths['importancia'] = 'images/importancia_variaveis.png'
        
        # Reais vs Previstos
        plot_reais_vs_previstos(y_test, y_pred, os.path.join(app.config['UPLOAD_FOLDER'], 'reais_vs_previstos.png'))
        plot_paths['reais_vs_previstos'] = 'images/reais_vs_previstos.png'
        
        # Resíduos
        plot_residuos(y_test, y_pred, os.path.join(app.config['UPLOAD_FOLDER'], 'residuos.png'))
        plot_paths['residuos'] = 'images/residuos.png'
        
        # 6. Renderizar template com resultados
        return render_template('resultados.html',
                             r2=r2,
                             rmse=rmse,
                             mae=mae,
                             coeficientes=coeficientes.to_dict('records'),
                             vif=vif.to_dict('records'),
                             model_summary=modelo_sm.summary().as_text(),
                             plots=plot_paths)
    
    except Exception as e:
        return render_template('error.html', error=str(e))
        
        # 5. Renderizar template com resultados
        return render_template('resultados.html',
                             r2=r2,
                             rmse=rmse,
                             mae=mae,
                             coeficientes=coeficientes.to_dict('records'),
                             vif=vif.to_dict('records'),
                             model_summary=modelo_sm.summary().as_text(),
                             plots=plot_paths)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

# Rota para servir arquivos estáticos
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)