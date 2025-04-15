#importando as bibliotecas python
from flask import Flask, render_template, request, redirect, flash, url_for, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt #Criptografar Senha
import secrets #gera números aleatórios criptograficamente fortes para gerenciar dados confidenciais

app = Flask(__name__)

#Chamando funções para o banco de dados
db = SQLAlchemy()
#Temos que pensar numa forma de ocultar os dados do banco no link
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqldb://univespsjc:Database2025@univespsjc.mysql.pythonanywhere-services.com:3306/univespsjc$default"
app.config['SQLALCHEMY_POOL_RECYCLE'] = 299
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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

#Definindo a pagina inicial
@app.route('/')
def index():
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
                return redirect(url_for('principal'))
            else:
                flash('Sua conta está desativada. Entre em contato com o suporte.', 'error')
                return redirect(url_for('index'))
        else:
            flash('Senha incorreta', 'error')
            return redirect(url_for('index'))

#Pagina Principal
@app.route('/principal')
def principal():
    if 'usuario_id' not in session:
        return redirect(url_for('index'))
    return render_template('principal.html')

#Logout do usuario
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))




