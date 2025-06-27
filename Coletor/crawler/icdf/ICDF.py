import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seus dados

df = pd.read_csv('threshold/AuthenticGames_sentiment.csv')

df_negative = df['negative']

list_negative = df_negative.tolist()

scores = np.array(sorted(list_negative))  # Garantir que os dados estejam ordenados

# Percentis
percentis = np.linspace(0, 1, len(scores))

# Aplicando o algoritmo de joelho
knee_locator = KneeLocator(percentis, scores, curve="concave", direction="increasing")
knee_percentil = knee_locator.knee
knee_valor = knee_locator.knee_y

# Plot
plt.plot(percentis, scores, label='ICDF')
plt.xlabel("Percentil")
plt.ylabel("Toxicidade (P(Negativo))")
plt.title("ICDF da Toxicidade")
plt.grid(True)

# Adiciona o ponto de joelho se existir
if knee_percentil is not None and knee_valor is not None:
    plt.axvline(knee_percentil, color='r', linestyle='--', label=f'Joelho: {knee_valor:.2f}')
    plt.scatter(knee_percentil, knee_valor, color='red', s=80)
    print(f"Ponto de joelho encontrado: percentil={knee_percentil}, toxicidade={knee_valor}")
else:
    print("Nenhum ponto de joelho foi encontrado.")

plt.legend()
plt.savefig("icdf/com_joelho.png", dpi=300, bbox_inches='tight')
plt.close()
