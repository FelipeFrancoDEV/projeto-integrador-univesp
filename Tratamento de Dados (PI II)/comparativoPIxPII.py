import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ========================================
# Dados comparativos
# ========================================
dados = pd.DataFrame({
    "Métrica": ["R²", "MAE", "RMSE"],
    "Semestre Passado (ElasticNet)": [0.5704, 20.28, 31.87],
    "Semestre Atual (Random Forest)": [0.9477, 5.58, 9.35]
})

# Transpor os dados para facilitar plot
dados_plot = dados.melt(id_vars="Métrica", var_name="Modelo", value_name="Valor")

# ========================================
# Configuração do gráfico
# ========================================
plt.figure(figsize=(8,5))
sns.barplot(data=dados_plot, x="Métrica", y="Valor", hue="Modelo", palette="Set2")

plt.title("Evolução do Desempenho do Modelo")
plt.ylabel("Valor da Métrica")
plt.ylim(0, max(dados_plot["Valor"])*1.1)  # deixa espaço no topo
plt.legend(title="")
plt.tight_layout()
plt.show()
