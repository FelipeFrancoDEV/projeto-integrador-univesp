import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Carregar dados
csv_path = r"C:\Users\Ana Clara Azevedo\Documents\Univesp (2ª Graduação)\5º Semestre\Projeto Integrador em Computação II\Quinzena 5\Dados Target.csv"
df = pd.read_csv(csv_path, sep=";", decimal=",")

# Ajustar colunas numéricas (remover espaços e converter para float)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(" ", "").astype(float)

# Selecionar apenas variáveis independentes
X = df.drop("Target", axis=1)

# Adicionar constante para statsmodels
X_const = sm.add_constant(X)

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variável"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print(vif_data)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dados do VIF
vif_data = {
    "Variável": ["const", "Input01", "Input02", "Input03", "Input04", "Input05", "Input06", "Input07", "Input08"],
    "VIF": [4854.717570, 7.770896, 7.681148, 1.422062, 1.392594, 12702.578332, 12701.577696, 1.482428, 1.556261]
}

df_vif = pd.DataFrame(vif_data)

# Destacar Inputs 5 e 6
colors = ["red" if x in ["Input05", "Input06"] else "blue" for x in df_vif["Variável"]]

plt.figure(figsize=(10,6))
sns.barplot(x="Variável", y="VIF", data=df_vif, palette=colors)
plt.title("VIF das Variáveis")
plt.ylabel("Valor do VIF")
plt.xlabel("Variável")
plt.axhline(10, color='gray', linestyle='--', label='VIF Limite (10)')
plt.legend()
plt.tight_layout()
plt.show()
