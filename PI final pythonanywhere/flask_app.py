#!/home/univespsjc/.virtualenvs/env/bin/python
#importando as bibliotecas python
import os
from flask import Flask, render_template, request, redirect, flash, url_for, session, current_app
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import bcrypt #Criptografar Senha
import secrets #gera números aleatórios criptograficamente fortes para gerenciar dados confidenciais
import traceback
from regressao import (
    diagnosticar_dados,
    preprocessar_dados,
    treinar_modelo_elasticnet,
    avaliar_modelo,
    calcular_importancia_features_pca,
    obter_coeficientes_modelo, # Adicionado para obter coeficientes do modelo
    # Funções de plotagem refatoradas
    plot_correlacao,
    plot_distribuicao,
    plot_importancia,
    plot_reais_vs_previstos,
    plot_distribuicao_residuos,
    plot_qq_residuos
)
from datetime import timedelta
# Importar funções refatoradas do módulo regressao

app = Flask(__name__, template_folder="templates", static_folder="static")



#Chamando funções para o banco de dados
db = SQLAlchemy()
#Temos que pensar numa forma de ocultar os dados do banco no link
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqldb://univespsjc:Database2025@univespsjc.mysql.pythonanywhere-services.com:3306/univespsjc$default"
app.config['SQLALCHEMY_POOL_RECYCLE'] = 3600
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False #melhora performance desativando um recurso não essencial
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle': 3600,'pool_pre_ping': True} #Verifica se a conexão está ativa antes de usar e Recicla a conexão a cada 3600s
app.config["PLOT_FOLDER"] = os.path.join(app.static_folder, "images")
db.init_app(app)#Estabelecendo conexão

app.secret_key = secrets.token_hex(32) #gerando chave secreta para troca de mensagens e metodo redirect

# Definindo uma classe que representa a tabela 'usuarios' no banco de dados
# Esta classe herda de db.Model, que é a base para todas as classes de modelo no SQLAlchemy
class usuarios(db.Model):
    __tablename__ = 'usuarios'  # Nome da tabela no banco de dados

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    usuario = db.Column(db.String(50), unique=True, nullable=False)
    senha = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    data_criacao = db.Column(db.DateTime, server_default=db.func.current_timestamp())
    ativo = db.Column(db.Boolean, nullable=False, default=0) # 0=inativo, 1=ativo

    @property
    def esta_ativo(self):
        #Retorna True se o usuário está ativo (1)
        return self.ativo == 1

class Dados(db.Model):
    __tablename__ = 'dados'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    target = db.Column(db.Numeric(10, 2))
    input01 = db.Column(db.Numeric(10, 2))
    input02 = db.Column(db.Numeric(10, 2))
    input03 = db.Column(db.Numeric(10, 2))
    input04 = db.Column(db.Numeric(10, 2))
    input05 = db.Column(db.Numeric(10, 2))
    input06 = db.Column(db.Numeric(10, 2))
    input07 = db.Column(db.Numeric(10, 2))
    input08 = db.Column(db.Numeric(10, 2))

    def to_dict(self):
        return {
            'target': float(self.target) if self.target is not None else None,
            'input01': float(self.input01) if self.input01 is not None else None,
            'input02': float(self.input02) if self.input02 is not None else None,
            'input03': float(self.input03) if self.input03 is not None else None,
            'input04': float(self.input04) if self.input04 is not None else None,
            'input05': float(self.input05) if self.input05 is not None else None,
            'input06': float(self.input06) if self.input06 is not None else None,
            'input07': float(self.input07) if self.input07 is not None else None,
            'input08': float(self.input08) if self.input08 is not None else None
        }


def carregar_dados():
    registros = Dados.query.all()

    # Converte cada registro para um dicionário e cria um DataFrame
    dados = pd.DataFrame([registro.to_dict() for registro in registros])

    return dados

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario_id' not in session:
            flash('Sua sessão expirou ou você precisa fazer login', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

#Definindo a pagina inicial
@app.route('/')
def index():
    # Verifica se o usuário está logado (se existe 'usuario_id' na sessão)
    if 'usuario_id' in session:
        return redirect(url_for('principal'))  # Redireciona para a página principal
    # Se não estiver logado, mostra a página index normalmente
    return render_template('index.html')

#definindo o metodos para o formulário de Cadastro
@app.route('/cadastro', methods=['POST'])
def cadastro():
    #Pegando os dados dos campos do Formulário
    usuario = request.form['usuario']
    email = request.form['email']
    senha = request.form['senha'].encode('utf-8')

    # Verificar se usuário já existe
    usuario_existente = usuarios.query.filter(
        (usuarios.usuario == usuario) |
        (usuarios.email == email)
        ).first()
    #definindo as mensagens que serao exibidas para o usuario
    if usuario_existente:
        if usuario_existente.usuario == usuario:
            flash('Nome de usuário já está em uso', 'error')

        else:
            flash('Email já está cadastrado', 'error')

        #retornando a pagina inicial
        return redirect(url_for('index'))

    #Criptografando a senha e inserindo os dados no Banco
    senha_hash = bcrypt.hashpw(senha, bcrypt.gensalt())
    senha_hash = senha_hash.decode('utf-8')
    insert = usuarios(usuario=usuario, senha=senha_hash, email=email)
    db.session.add(insert)
    db.session.commit()

    #Retorna a pagina de confirmação do Cadastro
    return render_template('alert_cad.html')

#Metodo para Login
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        user = request.form['usuario']
        senha = request.form['senha'].encode('utf-8')

        # Verificar se é email ou nome de usuário
        if '@' in user:
            usuario = usuarios.query.filter_by(email=user).first()
        else:
            usuario = usuarios.query.filter_by(usuario=user).first()

        if not usuario:
            flash('Usuário ou email não encontrado', 'error')
            return redirect(url_for('index'))

        # Verificar senha com bcrypt
        if bcrypt.checkpw(senha, usuario.senha.encode('utf-8')):
            if usuario.esta_ativo:
                # Login bem-sucedido
                session['usuario_id'] = usuario.id
                session['usuario_nome'] = usuario.usuario
                session.permanent = True
                app.permanent_session_lifetime = timedelta(minutes=5) # Expira em 30 minutos
                return redirect(url_for('principal'))
            else:
                flash('Sua conta está desativada. Entre em contato com o suporte.', 'error')
                return redirect(url_for('index'))
        else:
            flash('Senha incorreta', 'error')
            return redirect(url_for('index'))

#Pagina Principal
@app.route('/principal')
@login_required
def principal():
    if 'usuario_id' not in session:
        return redirect(url_for('index'))
    return render_template('principal.html')

@app.route("/analisar")
def analisar():


    """Executa a análise de regressão completa e renderiza os resultados."""
    try:
        # 1. Carregar Dados
        dados = carregar_dados()
        if dados is None:
            raise FileNotFoundError(f"Arquivo de dados {app.config['DATA_FILE']} não encontrado ou inválido.") # Correção de sintaxe

        # 2. Diagnóstico Inicial
        diagnostico = diagnosticar_dados(dados, 'target')
        if diagnostico is None:
            available_cols = dados.columns.tolist()
            head=dados.head()
            raise ValueError(f"Coluna alvo '{target}' não encontrada. Colunas disponíveis: {available_cols} Dados: {head}")
        vif_df = diagnostico["vif_original"]
        feature_names_original = diagnostico["feature_names_original"]

        # 3. Pré-processamento (Aplicando PCA devido ao alto VIF diagnosticado)
        target='target'
        dados_preparados = preprocessar_dados(
            dados,
            target,
            aplicar_pca=True,
            padronizar=True,
            n_components=6, # Usando 6 componentes como na análise anterior
            random_state=42
        )
        if dados_preparados is None:
            raise ValueError("Falha no pré-processamento dos dados.")

        # 4. Treinar Modelo (ElasticNet Otimizado)
        modelo = treinar_modelo_elasticnet(
            dados_preparados["X_train"],
            dados_preparados["y_train"],
            random_state=42
        )

        # 5. Avaliar Modelo
        y_pred, metricas = avaliar_modelo(
            modelo,
            dados_preparados["X_test"],
            dados_preparados["y_test"]
        )

        # 6. Interpretação e Coeficientes
        importancia_df = None
        coeficientes_df = None
        variancia_pca_df = None

        if dados_preparados.get("pca"):
            importancia_df = calcular_importancia_features_pca(
                dados_preparados["pca"],
                modelo,
                feature_names_original
            )
            variancia_pca_df = dados_preparados["variancia_pca"]
            coeficientes_df = obter_coeficientes_modelo(modelo, dados_preparados["feature_names"]) # PCs
        else:
            coeficientes_df = obter_coeficientes_modelo(modelo, dados_preparados["feature_names"])

        # 7. Gerar Visualizações
        plot_paths = {}
        plot_folder = app.config["PLOT_FOLDER"]

        plots_a_gerar = {
            "correlacao": (plot_correlacao, {"df": dados}),
            "distribuicao": (plot_distribuicao, {"serie": dados[target], "var_name": target}),
            "importancia": (plot_importancia, {"importancia_df": importancia_df}),
            "reais_vs_previstos": (plot_reais_vs_previstos, {"y_test": dados_preparados["y_test"], "y_pred": y_pred}),
            "residuos": (plot_distribuicao_residuos, {"y_test": dados_preparados["y_test"], "y_pred": y_pred}),
            "Q_Qplot": (plot_qq_residuos, {"y_test": dados_preparados["y_test"], "y_pred": y_pred}),
        }

        for key, (plot_func, kwargs) in plots_a_gerar.items():
            save_path = os.path.join(plot_folder, f"{key}.png")
            kwargs_com_path = {**kwargs, "save_path": save_path}
            try:
                plot_func(**kwargs_com_path)
                plot_paths[key] = url_for("static", filename=f"images/{key}.png")
            except Exception as plot_error:
                app.logger.error(f"Erro ao gerar gráfico '{key}': {plot_error}") # Melhoria no log
                plot_paths[key] = None

        # 8. Preparar Dados para Template
        vif_list = vif_df.to_dict("records") if vif_df is not None else []
        variancia_list = variancia_pca_df.to_dict("records") if variancia_pca_df is not None else []
        importancia_list = importancia_df.to_dict("records") if importancia_df is not None else []

        # 9. Renderizar Resultados
        return render_template(
            "resultados.html",
            r2=metricas.get("r2", None),
            rmse=metricas.get("rmse", None),
            mae=metricas.get("mae", None),
            vif=vif_list,
            variancia=variancia_list,
            importancia=importancia_list,
            plots=plot_paths
        )

    except FileNotFoundError as e:
        app.logger.error(f"Erro de arquivo: {e}")
        return render_template("error.html", error_title="Erro ao Carregar Dados", error_message=str(e))
    except ValueError as e:
        app.logger.error(f"Erro de valor: {e}")
        return render_template("error.html", error_title="Erro nos Dados ou Configuração", error_message=str(e))
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Erro inesperado durante a análise: {e}\n{error_trace}")
        return render_template("error.html", error_title="Erro Inesperado", error_message="Ocorreu um erro inesperado durante a análise. Por favor, verifique os logs do servidor.")


# Rota de Erro
@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error_title="Página Não Encontrada", error_message="A página que você está procurando não existe."), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Erro interno do servidor: {e}\n{traceback.format_exc()}")
    return render_template("error.html", error_title="Erro Interno do Servidor", error_message="Ocorreu um erro interno no servidor."), 500

#Logout do usuario
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))
