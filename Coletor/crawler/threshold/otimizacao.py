import pandas as pd
import numpy as np
from scipy.stats import expon
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from rich.console import Console

console = Console()

BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_TOXICIDADE = 'toxicity'

# Configurações da análise
LIMITE_INFERIOR_GZ = 0.30 

# Limites Inferiores (Início da GZ) a testar: de 0.05 a 0.35
RANGE_GZ_LOW = np.arange(0.05, 0.40, 0.05) 

# Limites Superiores (Início de T) a testar: de 0.40 a 0.95
RANGE_T_HIGH = np.arange(0.40, 1.0, 0.05) 

# Limites Superiores (Início de T) a testar: Níveis de confiança da Exponencial
NIVEIS_CONFIANCA_TESTE = np.concatenate([
    np.arange(0.80, 0.99, 0.01),
    np.arange(0.99, 0.999, 0.001)
])

'''
    Função para carregar os dados de toxicidade de uma lista de youtubers
    @param youtubers_lista - Lista de youtubers a serem analisados
    @return np.ndarray - Lista de listas de toxicidade de cada vídeo
'''
def carregar_dados_toxicidade(youtubers_list: list) -> np.ndarray:
    console.print("[bold]Carregando dados brutos de toxicidade...[/bold]")
    todos_scores = []
    
    for youtuber in youtubers_list:
        path = BASE_FOLDER / youtuber
        if not path.is_dir(): continue
        
        for file in path.rglob(INPUT_FILENAME):
            try:
                df = pd.read_csv(file)
                if COLUNA_TOXICIDADE in df.columns:
                    todos_scores.extend(df[COLUNA_TOXICIDADE].dropna().tolist())
            except:
                pass
    
    return np.array(todos_scores).reshape(-1, 1)

'''
    Função para classificar os dados em 3 clusters baseado em UM threshold
    @param dados - Lista de listas de toxicidade de cada vídeo
    @param threshold_t - Threshold calculado pela exponencial
'''
def calcular_silhouette_score_1d(dados: np.ndarray, threshold_t: float) -> float:
    # Se o threshold calculado for menor que o limite da zona cinza, a lógica quebra
    if threshold_t <= LIMITE_INFERIOR_GZ:
        return -1 

    labels = np.zeros(dados.shape[0])
    
    # Label 0: NT ( < 0.30 ) -> Já é 0 por padrão
    
    # Label 1: GZ ( >= 0.30 e < threshold_t )
    mask_gz = (dados >= LIMITE_INFERIOR_GZ) & (dados < threshold_t)
    labels[mask_gz.ravel()] = 1
    
    # Label 2: T ( >= threshold_t )
    mask_t = (dados >= threshold_t)
    labels[mask_t.ravel()] = 2
    
    # Validação de clusters mínimos
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1 
        
    return silhouette_score(dados, labels)

'''
    Função para classificar os dados em 3 clusters baseados nos DOIS thresholds
    @param dados - Lista de listas de toxicidade
    @param threshold_gz_low - Limite inferior da Zona Cinza
    @param threshold_t_high - Limite inferior da Toxicidade (calculado via exponencial)
'''
def calcular_silhouette_score_2d(dados: np.ndarray, threshold_gz_low: float, threshold_t_high: float) -> float:
    # Validação básica: T deve ser maior que GZ
    if threshold_t_high <= threshold_gz_low:
        return -1.0

    labels = np.zeros(dados.shape[0])
    
    # Label 0: NT ( < GZ_low )
    
    # Label 1: GZ ( >= GZ_low e < T_high )
    mask_gz = (dados >= threshold_gz_low) & (dados < threshold_t_high)
    labels[mask_gz.ravel()] = 1
    
    # Label 2: T ( >= T_high )
    mask_t = (dados >= threshold_t_high)
    labels[mask_t.ravel()] = 2
    
    # Validação de clusters mínimos (precisa ter pelo menos 2 grupos populados)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
   
    return silhouette_score(dados, labels)


'''
    Função orquestradora para otimizar a escolha do threshold de toxicidade
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def otimizar_threshold_silhouette_score(youtubers_list: list):
    # Carregar os dados de toxicidade dos youtubers
    dados = carregar_dados_toxicidade(youtubers_list)

    # Testa se os dados foram encontrados
    if len(dados) == 0:
        console.print("[red]Nenhum dado encontrado.[/red]")
        return

    console.print(f"Dados carregados: {len(dados)} amostras.")
    console.print(f"Fixando Limite Inferior (NT/GZ) em: {LIMITE_INFERIOR_GZ}")
    console.print("Testando Threshold Superior (GZ/T)...")

    scores = []
    for t in RANGE_T_HIGH:
        score = calcular_silhouette_score_1d(dados, t)
        scores.append(score)
        console.print(f" Threshold T={t:.2f} -> Silhouette: {score:.4f}")


    # Encontrar o melhor silhouette Index
    best_idx = np.argmax(scores)
    best_threshold = RANGE_T_HIGH[best_idx]
    best_score = scores[best_idx]

    console.print(f"\n[bold green]Melhor Threshold Encontrado: {best_threshold:.2f}[/bold green]")
    console.print(f"Silhouette Index: {best_score:.4f}")

    # Plotar
    plt.figure(figsize=(10, 6))
    plt.plot(RANGE_T_HIGH, scores, marker='o')
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'Melhor: {best_threshold:.2f}')
    plt.title(f'Otimização de Threshold de Toxicidade (Silhouette Index)\nBaseado em {len(dados)} tiras')
    plt.xlabel('Threshold Superior (Início da Toxicidade)')
    plt.ylabel('Silhouette Index')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico
    output_path = BASE_FOLDER / 'threshold'
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / 'otimizacao_silhouette_score.png'

    plt.savefig(output_path)
    plt.close()

    console.print(f"Gráfico salvo em: {output_path}") 

'''
    Função orquestradora para otimizar AMBOS os thresholds (GZ e T) de toxicidade
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def otimizar_dois_thresholds_silhouette_score(youtubers_list: list):
    console.print("\n[bold magenta]=== Otimização 2D: Nível GZ + Nível T ===[/bold magenta]")
    
    # Carregar Dados
    dados = carregar_dados_toxicidade(youtubers_list)
    if len(dados) == 0:
        console.print("[red]Nenhum dado encontrado.[/red]")
        return

    console.print(f"Dados carregados: {len(dados)} amostras.")

    # Matriz para armazenar os resultados do Grid Search
    # Linhas (i) = GZ_low, Colunas (j) = T_high
    resultados_grid = np.zeros((len(RANGE_GZ_LOW), len(RANGE_T_HIGH)))
    
    melhor_score = -1
    melhor_config = None

    console.print("Executando busca em grade (Grid Search)...")

    # Loop 2D
    for i, gz_low in enumerate(RANGE_GZ_LOW):
        for j, t_high in enumerate(RANGE_T_HIGH):
            # T deve ser estritamente maior que GZ para formar 3 clusters. Se não, é uma configuração inválida.
            if t_high <= gz_low:
                resultados_grid[i, j] = -1 # Marcamos como inválido
                continue

            # Calcula o Silhouette Index para o par (gz_low, t_high)
            score = calcular_silhouette_score_2d(dados, gz_low, t_high)
            resultados_grid[i, j] = score
            
            # Verifica se é o melhor até agora
            if score > melhor_score:
                melhor_score = score
                melhor_config = {
                    'gz_low': gz_low,
                    't_high': t_high
                }

    # Resultados
    if melhor_config:
        console.print(f"\n[bold green]RESULTADO DA OTIMIZAÇÃO 2D:[/bold green]")
        console.print(f"Melhor Silhouette Index: {melhor_score:.4f}")
        console.print(f"Melhor Limite Inferior (GZ): [bold]{melhor_config['gz_low']:.2f}[/bold]")
        console.print(f"Melhor Limite Superior (T): [bold]{melhor_config['t_high']:.4f}[/bold]")
        console.print(f"Intervalos Sugeridos: NT < {melhor_config['gz_low']:.2f} <= GZ < {melhor_config['t_high']:.2f} <= T")
    else:
        console.print("[red]Não foi possível encontrar uma configuração válida.[/red]")
        return

    # Plotagem do gráfico
    plt.figure(figsize=(12, 10))
    
    # Preparar labels formatados para os eixos (apenas 2 casas decimais)
    x_labels = [f"{val:.2f}" for val in RANGE_T_HIGH]
    y_labels = [f"{val:.2f}" for val in RANGE_GZ_LOW]
    
    # Criar uma máscara para esconder os valores inválidos (onde score é -1 ou 0 por definição)
    mask = resultados_grid <= 0
    
    ax = sns.heatmap(
        resultados_grid, 
        xticklabels=x_labels, 
        yticklabels=y_labels,
        mask=mask,          # Esconde dados inválidos
        #cmap='viridis',     # Cores intensas (amarelo = melhor score)
        annot=True,         # Mostra os valores
        fmt=".2f",          # Formata com 2 casas decimais
        cbar_kws={'label': 'Silhouette Index'},
        linewidths=.5,      # Linhas brancas para separar as células
        linecolor='white'
    )
    
    # Ajuste fino dos eixos
    plt.title(f'Otimização 2D de Thresholds (Silhouette Index)\nMelhor Silhouette Index: {melhor_score:.4f} \nIntervalos Sugeridos: NT < {melhor_config['gz_low']:.2f} <= GZ < {melhor_config['t_high']:.2f} <= T', fontsize=16, pad=20)
    plt.xlabel('Limite Superior da Toxicidade (T)', fontsize=12) # Eixo X
    plt.ylabel('Limite Inferior da Zona Cinza (GZ)', fontsize=12) # Eixo Y
    
    # Inverter o eixo Y para que os valores menores de GZ fiquem embaixo (padrão cartesiano)
    # ou manter em cima (padrão de matriz). O padrão de matriz (0 no topo) é o default do heatmap.
    # Se quiser cartesiano (0 embaixo), descomente a linha abaixo:
    ax.invert_yaxis()

    plt.tight_layout()
    
    output_path = BASE_FOLDER / 'threshold'
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'otimizacao_2d_heatmap.png'
    
    plt.savefig(output_file, dpi=300)
    plt.close()

    console.print(f"Heatmap de otimização 2D salvo em: [green]{output_file}[/green]")


'''
    Função orquestradora para otimizar o Nível de Confiança da Distribuição Exponencial
    baseado na melhor separação de clusters (Silhouette Index)
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def otimizar_confianca_exponencial(youtubers_list: list):
    console.print("\n[bold magenta]=== Otimização Híbrida: Exponencial + Silhouette ===[/bold magenta]")
    
    # Carregar Dados
    dados = carregar_dados_toxicidade(youtubers_list)
    if len(dados) == 0:
        console.print("[red]Nenhum dado encontrado.[/red]")
        return

    console.print(f"Dados carregados: {len(dados)} amostras.")
    
    # Ajusta a distribuição exponencial e encontra o lambda (scale = 1/lambda) que melhor descreve os dados
    loc, scale = expon.fit(dados, floc=0)
    console.print(f"Parâmetros Exponencial: Scale (1/lambda) = {scale:.4f}")

    console.print("Testando níveis de confiança...")
    
    resultados_confianca = []
    resultados_threshold = []
    resultados_silhouette = []

    # Loop de otimização
    for confianca in NIVEIS_CONFIANCA_TESTE:
        # Calcula o threshold estatístico para esta confiança
        # ppf = Percentile Point Function (Inverso da CDF)
        threshold_calc = expon.ppf(confianca, loc=loc, scale=scale)
        
        # Ignora thresholds absurdos (maiores que 1.0 ou menores que GZ)
        if threshold_calc > 1.0 or threshold_calc <= LIMITE_INFERIOR_GZ:
            continue

        # Calcula a qualidade da separação com este threshold
        score = calcular_silhouette_score_1d(dados, threshold_calc)
        
        resultados_confianca.append(confianca)
        resultados_threshold.append(threshold_calc)
        resultados_silhouette.append(score)
        
    if not resultados_silhouette:
        console.print("[red]Nenhum nível de confiança gerou um threshold válido.[/red]")
        return

    # Encontrar o Melhor Resultado
    best_idx = np.argmax(resultados_silhouette)
    best_conf = resultados_confianca[best_idx]
    best_thresh = resultados_threshold[best_idx]
    best_sil = resultados_silhouette[best_idx]

    console.print(f"\n[bold green]RESULTADO DA OTIMIZAÇÃO:[/bold green]")
    console.print(f"Melhor Nível de Confiança: [bold]{best_conf:.1%}[/bold]")
    console.print(f"Threshold Resultante (T): [bold]{best_thresh:.4f}[/bold]")
    console.print(f"Silhouette Index: {best_sil:.4f}")

    # Plotar Gráficos
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo Y1 (Esquerda): Silhouette Index
    color = 'tab:blue'
    ax1.set_xlabel('Nível de Confiança Estatística (Percentil)')
    ax1.set_ylabel('Silhouette Index', color=color, fontsize=12)
    ax1.plot(resultados_confianca, resultados_silhouette, color=color, linewidth=2, label='Silhouette')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Destaque para o melhor ponto
    ax1.axvline(best_conf, color='green', linestyle='--', label=f'Melhor: {best_conf:.1%}')

    # Eixo Y2 (Direita): Threshold Resultante
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Threshold de Toxicidade Resultante', color=color, fontsize=12)
    ax2.plot(resultados_confianca, resultados_threshold, color=color, linestyle=':', alpha=0.7, label='Threshold')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Otimização do Nível de Confiança (Exponencial)\nMelhor Threshold: {best_thresh:.3f} (Confiança: {best_conf:.1%})')
    
    # Salvar
    output_path = BASE_FOLDER / 'threshold'
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'otimizacao_exponencial_hibrida.png'
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    console.print(f"Gráfico de otimização salvo em: [green]{output_file}[/green]")

if __name__ == "__main__":
    lista_youtubers = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer']
    
    #otimizar_threshold_silhouette_score(lista_youtubers)
    #otimizar_confianca_exponencial(lista_youtubers)
    otimizar_dois_thresholds_silhouette_score(lista_youtubers)