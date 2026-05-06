from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

def validar_non_toxic_cluster(df_vmg, algoritmo_cluster):
    cluster_col = f'Cluster_{algoritmo_cluster}'
    clusters = sorted(df_vmg[cluster_col].unique())

    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    
    non_toxic_count = np.zeros(len(clusters))
    quantidade_toxico = df_vmg['NT->T'] + df_vmg['GZ->T'] + df_vmg['T->T']

    for i, video in enumerate(quantidade_toxico):
        if video == 0:
            cluster = df_vmg.iloc[i][cluster_col]
            idx = cluster_to_idx[cluster]
            non_toxic_count[idx] += 1

    total_non_toxic = non_toxic_count.sum()

    for cluster_id, idx in cluster_to_idx.items():
        cluster_df = df_vmg[df_vmg[cluster_col] == cluster_id]
        cluster_size = len(cluster_df)
        count = non_toxic_count[idx]


        if cluster_size > 0:
            percentage = count / cluster_size * 100
        else:
            percentage = 0

        global_ratio = (count / total_non_toxic * 100) if total_non_toxic > 0 else 0

        console.print(f'Cluster {cluster_id}:'
                      f'\nProporção Non-toxic dentro do Cluster = {count} / {cluster_size} [{percentage:.2f}%]'
                      f'\nProporção Non-toxic geral = {count} / {total_non_toxic} [{global_ratio:.2f}%]'
        )
        console.print(f"------------------------------------------------------------------")


def gerar_grafico_influencia(df_vmg, algoritmo_cluster, toxicidade_cluster):
    cluster_col = f'Cluster_{algoritmo_cluster}'

    df_vmg['quantidade_t'] = df_vmg['NT->T'] + df_vmg['GZ->T'] + df_vmg['T->T']
    df_vmg['quantidade_nt'] = df_vmg['NT->NT'] + df_vmg['GZ->NT'] + df_vmg['T->NT']
    df_vmg['quantidade_gz'] = df_vmg['NT->GZ'] + df_vmg['GZ->GZ'] + df_vmg['T->GZ']

    metrics = [
        ("Tóxica", "quantidade_t"),
        ("Não-Tóxica", "quantidade_nt"),
        ("Gray-Zone", "quantidade_gz")
    ]

    clusters = sorted(df_vmg[cluster_col].unique())

    fig, axes = plt.subplots(
        nrows=len(clusters),
        ncols=3,
        figsize=(18, 4 * len(clusters)),
        sharex=False,
        sharey=False
    )


    if len(clusters) == 1:
        axes = [axes]

    for i, cluster in enumerate(clusters):
        subset = df_vmg[df_vmg[cluster_col] == cluster]

        for j, (label, col) in enumerate(metrics):
            ax = axes[i][j] if len(clusters) > 1 else axes[j]

            data = subset[col]

            ax.hist(data, bins=range(int(data.max()) + 2), edgecolor='black')

            ax.set_title(label if i == 0 else "")
            ax.set_ylabel(f'Cluster {cluster}' if j == 0 else "")
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Análise dos Clusters. Quantidade de tirinhas por quantidade de visitas em cada nó ({algoritmo_cluster} | {toxicidade_cluster})',
        fontsize=16,
        y=0.99
    )

    # plt.tight_layout()
    plt.show()



algoritmo = ['KMeans','DBSCAN']
toxicidade = ['perspective','detoxify']

for t in toxicidade:
    for a in algoritmo:
        console.print(f"[purple]== Testando Non-Toxic para {a} | {t} ==")
        csv_path = Path('files/VMG/Geral/Agrupamento') / f'cluster_{t}_contagem_{a.lower()}.csv'
        df = pd.read_csv(csv_path)
        validar_non_toxic_cluster(df,a)
        # gerar_grafico_influencia(df,a,t)