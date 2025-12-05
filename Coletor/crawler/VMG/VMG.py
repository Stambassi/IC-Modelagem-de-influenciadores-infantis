from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

console = Console()

# Configuração estética global para gráficos (padrão acadêmico)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
sns.set_context("paper", font_scale=1.2)

# Define as regras para cada tipo de análise (métrica)
METRICAS_CONFIG = {
    'sentimento': {
        'coluna_base': 'sentimento_dominante', 
        'tipo_estados': 'categorico', 
        'estados': ['POS', 'NEU', 'NEG'],
        # Cores para o grafo: Verde (POS), Cinza (NEU), Vermelho (NEG)
        'cores_grafo': {'POS': '#2ecc71', 'NEU': '#95a5a6', 'NEG': '#e74c3c'} 
    },
    'negatividade': {
        'coluna_base': 'negatividade',
        'tipo_estados': 'numerico',
        'n_estados': 3 
    },
    'toxicidade': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', 
        'limiares': [0.0, 0.20, 0.50, 1.01], 
        'estados': ['NT', 'GZ', 'T'],
        # Cores para o grafo: Verde (NT), Amarelo (GZ), Vermelho (T)
        'cores_grafo': {'NT': '#2ecc71', 'GZ': '#f1c40f', 'T': '#e74c3c'} 
    }
}

'''
    Função para armazenar as transições de estados (para uma métrica específica) de cada vídeo em um arquivo CSV
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3')
'''
def salvar_transicoes_por_metrica(youtubers_list: list[str], metrica_config: dict, nome_analise: str) -> None:
    # Extrair configurações da métrica
    coluna_base = metrica_config['coluna_base']
    tipo_estados = metrica_config['tipo_estados']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando transições de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan]')

        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            video_path = tiras_csv_path.parent
            try:
                (video_path / 'transicoes').mkdir(parents=True, exist_ok=True)
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se o DataFrame tem a coluna necessária e dados suficientes
                if df_tiras_video.empty or coluna_base not in df_tiras_video.columns or len(df_tiras_video) < 2:
                    continue

                # Lógica de Definição de Estado
                if tipo_estados == 'categorico':
                    estados = metrica_config['estados']

                    # Renomeia a coluna base (ex: 'sentimento_dominante') para 'estado'
                    df_tiras_video.rename(columns={coluna_base: 'estado'}, inplace=True)

                    # Filtra para garantir que apenas os estados definidos sejam usados
                    df_tiras_video = df_tiras_video[df_tiras_video['estado'].isin(estados)]

                    if len(df_tiras_video) < 2: continue

                    # Converte para tipo Categórico para garantir todas as transições
                    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)
                
                elif tipo_estados == 'numerico':
                    n = metrica_config['n_estados']
                    grupos = np.linspace(0, 1, n + 1)
                    labels = range(1, n + 1)

                    # Usar pd.cut para classificar cada valor da métrica em um estado
                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=grupos,
                        labels=labels,
                        include_lowest=True
                    )

                elif tipo_estados == 'numerico_categorizado':
                    # Usa os limiares e nomes de estados da configuração
                    bins = metrica_config['limiares']
                    labels = metrica_config['estados']

                    # Usar pd.cut para classificar cada valor da métrica em um estado
                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=bins,
                        labels=labels,
                        right=False, # Define intervalos como [a, b)
                        include_lowest=True # Garante que 0.0 seja incluído
                    )

                    # Converte para tipo Categórico
                    tipo_categorico = CategoricalDtype(categories=labels, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)

                    # Re-verificar se há dados suficientes após a filtragem
                    if len(df_tiras_video.dropna(subset=['estado'])) < 2:
                        continue

                else:
                    console.print(f"[red]Erro: Tipo de estado '{tipo_estados}' não reconhecido.[/red]")
                    continue

                # Lógica de transição (comum a ambos os tipos)
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa e conta, incluindo as transições que não ocorreram (contagem 0)
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')
                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Salva no novo arquivo CSV com nome descritivo
                output_path = video_path / 'transicoes' / f'transicoes_{nome_analise}.csv'
                contagem.to_csv(output_path, index=False)
            
            except Exception as e:
                console.print(f'[bold red]Erro[/bold red] em {video_path.name} (salvar_transicoes_por_metrica): {e}')

'''
    Função para criar e persistir a Matriz de Transição (VMG) para cada vídeo individual
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3')
'''
def salvar_matriz_transicao_video(youtubers_list: list[str], metrica_config: dict, nome_analise: str) -> None:
    # Definir os estados com base no tipo de métrica
    if metrica_config['tipo_estados'] == 'categorico' or metrica_config['tipo_estados'] == 'numerico_categorizado':
        estados = metrica_config['estados'] # Ex: ['POS','NEU','NEG'] ou ['NT','GZ','T']
    else: # numerico
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1))
    
    # Criar o tipo Categórico para garantir a forma da matriz (NxN)
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)

    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes de vídeo de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan]')

        # Procurar recursivamente pelo arquivo de transições de sentimento
        for transicoes_csv_path in base_path.rglob(f'transicoes_{nome_analise}.csv'):                
            try:                
                df_transicoes = pd.read_csv(transicoes_csv_path)
                if df_transicoes.empty:
                    continue

                # Garantir que a matriz final seja sempre NxN
                df_transicoes['estado'] = df_transicoes['estado'].astype(tipo_categorico)
                df_transicoes['proximo_estado'] = df_transicoes['proximo_estado'].astype(tipo_categorico)

                # Calcular a soma das transições que saem de cada estado
                somas_por_estado = df_transicoes.groupby('estado', observed=False)['contagem'].transform('sum')

                # Calcular a probabilidade de cada transição 
                probabilidade = (df_transicoes['contagem'] / somas_por_estado).fillna(0)
                df_transicoes['probabilidade'] = probabilidade

                # Transforma o formato "longo" para o formato "largo" (matriz)
                matriz_transicao = df_transicoes.pivot(
                    index='estado', 
                    columns='proximo_estado', 
                    values='probabilidade'
                )
                
                # Garantir que sejam 0 (caso algum estado nunca seja ponto de partida)
                matriz_transicao.fillna(0, inplace=True)
                
                # Salvar a matriz
                output_folder = transicoes_csv_path.parent.parent / 'VMG'
                output_folder.mkdir(parents=True, exist_ok=True)
                output_path = output_folder / f'VMG_{nome_analise}.csv'
                matriz_transicao.to_csv(output_path)

            except Exception as e:
                console.print(f'Inválido (salvar_matriz_transicao_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição agregada para cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados.
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG).
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3').
    @param agg_metrica - Tipo de cálculo da agregação (ex: 'mean', 'standard', 'variation').
'''
def salvar_matriz_transicao_youtuber(youtubers_list: list[str], metrica_config: dict, nome_analise: str, agg_metrica: str = 'mean') -> None:
    # Definir os estados com base no tipo de métrica
    if metrica_config['tipo_estados'] == 'categorico' or metrica_config['tipo_estados'] == 'numerico_categorizado':
        estados = metrica_config['estados'] # Ex: ['POS','NEU','NEG'] ou ['NT','GZ','T']
    else: # numerico
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1)) # Ex: [1, 2, 3]
    
    # Criar o tipo Categórico para garantir a ordem e completude da matriz
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
    
    # Mapeamento de funções auxiliares para dispersão
    agg_funcs = {
        'standard': 'std',
        'variation': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }

    if agg_metrica != 'mean' and agg_metrica not in agg_funcs:
        console.print(f"[bold red]Erro: Métrica de agregação '{agg_metrica}' é inválida.[/bold red]")
        return

    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue
        console.print(f'>>> Processando matriz agregada de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan] (Agregação: {agg_metrica})')

        try:
            # Encontrar e concatenar todas as transições do youtuber
            lista_dfs_transicoes_por_video = []
            for transicoes_csv_path in base_path.rglob(f'transicoes_{nome_analise}.csv'):
                df_video = pd.read_csv(transicoes_csv_path)
                if not df_video.empty:
                    # Tentar obter ID do vídeo, fallback para nome da pasta
                    try:
                        video_id_path = transicoes_csv_path.parent.parent / 'videos_info.csv'
                        df_video['video_id'] = pd.read_csv(video_id_path)['video_id'][0]
                    except Exception:
                         df_video['video_id'] = transicoes_csv_path.parent.parent.name
                    lista_dfs_transicoes_por_video.append(df_video)
            
            if not lista_dfs_transicoes_por_video:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições de '{nome_analise}' encontrado para {youtuber}. Pulando.[/yellow]")
                continue
            
            df_todas_contagens = pd.concat(lista_dfs_transicoes_por_video, ignore_index=True)

            # Garantir tipos categóricos
            df_todas_contagens['estado'] = df_todas_contagens['estado'].astype(tipo_categorico)
            df_todas_contagens['proximo_estado'] = df_todas_contagens['proximo_estado'].astype(tipo_categorico)
            
            if agg_metrica == 'mean':                
                # Agrupar por transição e somar contagens globais
                df_agregado = df_todas_contagens.groupby(['estado', 'proximo_estado'], observed=False)['contagem'].sum().reset_index()
                
                # Calcular o total de saídas de cada estado (Soma Global)
                somas_por_estado = df_agregado.groupby('estado', observed=False)['contagem'].transform('sum')
                
                # Calcular probabilidade (contagem global / soma global)
                df_agregado[agg_metrica] = (df_agregado['contagem'] / somas_por_estado).fillna(0)
                
            else:
                # Calcula a probabilidade dentro de cada vídeo para medir a variabilidade.               
                # Calcular totais por vídeo
                somas_por_estado_video = df_todas_contagens.groupby(['video_id', 'estado'], observed=False)['contagem'].transform('sum')
                
                # Probabilidade individual por vídeo
                df_todas_contagens['probabilidade'] = (df_todas_contagens['contagem'] / somas_por_estado_video).fillna(0)

                # Agregar estatisticamente (std, variation)
                agg_func = agg_funcs[agg_metrica]
                df_agregado = df_todas_contagens.groupby(['estado', 'proximo_estado'], observed=False)['probabilidade'].agg(agg_func).reset_index()
                df_agregado.rename(columns={'probabilidade': agg_metrica}, inplace=True)

            # Pivotar para formato de matriz
            matriz_transicao_youtuber = df_agregado.pivot(
                index='estado', 
                columns='proximo_estado', 
                values=agg_metrica
            )
            
            # Preencher NaNs com 0 e garantir formato
            matriz_transicao_youtuber.fillna(0, inplace=True)
            
            # Salvar
            output_folder = base_path / 'VMG'
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f'VMG_{nome_analise}_{agg_metrica}.csv'
            
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição ({agg_metrica}) salva em: {output_path}")

        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_youtuber): {e}')

'''
    Função para gerar um Heatmap (Mapa de Calor) com qualidade de publicação.
    Mostra as probabilidades de transição com anotações e escala de cor.

    @param matrix_path - Caminho para o arquivo CSV da matriz VTMG
    @param output_path - Caminho onde a imagem será salva
    @param title - Título do gráfico
'''
def gerar_heatmap_vmg(matrix_path: Path, output_path: Path, title: str):
    try:
        # Carrega a matriz definindo a primeira coluna como índice (Labels dos estados)
        df_matrix = pd.read_csv(matrix_path, index_col=0)
        
        # Cria a figura
        plt.figure(figsize=(8, 6))
        
        # Gera o Heatmap
        # cmap='Blues' ou 'Reds' são boas opções acadêmicas. 'YlGnBu' é muito usado também.
        # vmin=0 e vmax=1 garantem que a escala de cor seja sempre absoluta (0% a 100%)
        ax = sns.heatmap(
            df_matrix, 
            annot=True,       # Escreve os números nas células
            fmt=".2f",        # Formato 2 casas decimais
            cmap="Blues",      # Mapa de cor (Vermelho é bom para toxicidade)
            linewidths=.5,    # Linhas separando células
            linecolor='gray',
            vmin=0, vmax=1,   # Trava a escala entre 0 e 1
            cbar_kws={'label': 'Probabilidade de Transição'}
        )
        
        # Ajustes finos de layout
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel("Próximo Estado (t+1)", fontsize=12)
        ax.set_ylabel("Estado Atual (t)", fontsize=12)
        
        # Salva com alta resolução e margens ajustadas
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        console.print(f"[red]Erro ao gerar heatmap de {matrix_path.name}: {e}[/red]")

'''
    Função para gerar um Grafo Dirigido (NetworkX) visualizando as transições.
    A espessura da seta indica a probabilidade.

    @param matrix_path - Caminho para o arquivo CSV da matriz VTMG
    @param output_path - Caminho onde a imagem será salva
    @param config_cores - Dicionário mapeando nome do estado -> cor Hex
'''
def gerar_grafo_vmg(matrix_path: Path, output_path: Path, config_cores: dict = None):
    try:
        df_matrix = pd.read_csv(matrix_path, index_col=0)
        
        # Cria um grafo dirigido
        G = nx.DiGraph()
        
        # Adiciona os nós
        estados = df_matrix.index.tolist()
        G.add_nodes_from(estados)
        
        # Adiciona as arestas com pesos (probabilidades)
        for origem in estados:
            for destino in estados:
                peso = df_matrix.loc[origem, destino]

                G.add_edge(origem, destino, weight=peso)
        
        plt.figure(figsize=(8, 8))
        
        # Layout circular é ideal para matrizes pequenas (3x3)
        pos = nx.circular_layout(G)
        
        # Define cores dos nós
        node_colors = ['lightgray'] * len(G.nodes())
        if config_cores:
            node_colors = [config_cores.get(node, 'lightgray') for node in G.nodes()]
        
        # Desenha nós
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors='black')
        
        # Desenha labels dos nós
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Desenha arestas (setas)
        # Espessura baseada no peso * fator para ficar visível
        weights = [G[u][v]['weight'] * 4 for u, v in G.edges()]
        
        # connectionstyle='arc3, rad=0.1' faz as setas serem curvas
        # isso permite ver ida e volta (A->B e B->A) sem sobreposição
        nx.draw_networkx_edges(
            G, pos, 
            width=weights, 
            edge_color='gray', 
            arrowstyle='-|>', 
            arrowsize=20,
            connectionstyle='arc3, rad=0.2' 
        )
        
        # Adiciona labels nas arestas (opcional, pode poluir, mas professores gostam de dados)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=8)
        
        plt.title("Grafo de Probabilidade de Transição", fontsize=14)
        plt.axis('off') # Remove eixos cartesianos
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        console.print(f"[red]Erro ao gerar grafo de {matrix_path.name}: {e}[/red]")

'''
    Função nova que varre recursivamente todos os vídeos de cada youtuber
    e gera o gráfico individual para cada matriz de vídeo encontrada.
'''
def gerar_visualizacoes_individuais_videos(youtubers_list: list[str], nome_analise: str):
    console.print(f"\n[bold magenta]===== GERANDO HEATMAPS INDIVIDUAIS PARA '{nome_analise.upper()}' =====[/bold magenta]")
    
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): continue
        
        # Procura por arquivos de matriz VMG dentro das subpastas dos vídeos
        padrao_busca = f"VMG_{nome_analise}.csv"
        
        arquivos_vmg = list(base_path.rglob(padrao_busca))
        
        if not arquivos_vmg:
            console.print(f"     [yellow]Nenhuma matriz individual encontrada para {youtuber}.[/yellow]")
            continue
            
        console.print(f"   Gerando {len(arquivos_vmg)} gráficos para: {youtuber}")
        
        for vmg_file in arquivos_vmg:
            # Identifica o nome da pasta do vídeo para usar no título
            nome_video = vmg_file.parent.parent.name
            
            # Define onde salvar o plot (cria pasta 'plots' dentro da pasta do vídeo)
            pasta_plot = vmg_file.parent.parent / 'plots'
            pasta_plot.mkdir(exist_ok=True)
            
            output_file = pasta_plot / f'heatmap_{nome_analise}.png'
            
            titulo_grafico = f"VTMG (1ª Ordem): {nome_video}\nAnálise: {nome_analise.title()}"
            
            gerar_heatmap_vmg(vmg_file, output_file, titulo_grafico)

'''
    Função para gerar o Heatmap da Matriz Média Agregada do Youtuber
'''
def gerar_visualizacoes_agregadas(youtubers_list: list[str], nome_analise: str):
    console.print(f"\n[bold magenta]===== GERANDO HEATMAPS AGREGADOS PARA '{nome_analise.upper()}' =====[/bold magenta]")
    
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}/VMG')
        img_output_folder = Path(f'files/{youtuber}/plots')
        img_output_folder.mkdir(parents=True, exist_ok=True)
        
        matriz_media_path = base_path / f'VMG_{nome_analise}_mean.csv'
        
        if matriz_media_path.exists():
            heatmap_path = img_output_folder / f'heatmap_{nome_analise}_agregado.png'
            gerar_heatmap_vmg(
                matriz_media_path, 
                heatmap_path, 
                title=f"Matriz de Transição Média de {nome_analise}: {youtuber}"
            )
            console.print(f"   Agregado salvo para: {youtuber}")

'''
    Função principal para orquestrar o pipeline de análise de VMG
    @param youtubers_list - Lista de youtubers a serem analisados
    @param config_analise - Dicionário de configurações da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome identificador desta análise (ex: 'sentimento')
'''
def rodar_pipeline_vmg(youtubers_list: list[str], config_metrica: dict, nome_analise: str):
    console.print(f"\n[bold magenta]===== INICIANDO PIPELINE VMG PARA '{nome_analise.upper()}' =====[/bold magenta]")
    
    # Gerar arquivos de contagem de transições (por vídeo)
    salvar_transicoes_por_metrica(youtubers_list, config_metrica, nome_analise)
    
    # Gerar matrizes de probabilidade (por vídeo)
    salvar_matriz_transicao_video(youtubers_list, config_metrica, nome_analise)
    
    # Gerar matriz agregada (por youtuber) para diferentes agregações
    for agg in ['mean', 'standard', 'variation']:
        salvar_matriz_transicao_youtuber(youtubers_list, config_metrica, nome_analise, agg_metrica=agg)
    
    # Visualizações Individuais (Cada Vídeo)
    #gerar_visualizacoes_individuais_videos(youtubers_list, nome_analise)

    # Visualizações Agregadas (Por Youtuber)
    gerar_visualizacoes_agregadas(youtubers_list, nome_analise)

    console.print(f"\n[bold magenta]===== PIPELINE VSMG PARA '{nome_analise.upper()}' CONCLUÍDO =====[/bold magenta]")


if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    # # Executa o pipeline para a métrica 'sentimento'
    # rodar_pipeline_vmg(
    #     lista_youtubers, 
    #     config_metrica=METRICAS_CONFIG['sentimento'], 
    #     nome_analise='sentimento'
    # )
    
    # # Executa o pipeline para a métrica 'negatividade' com 3 estados
    # rodar_pipeline_vmg(
    #     lista_youtubers, 
    #     config_metrica=METRICAS_CONFIG['negatividade'], 
    #     nome_analise='negatividade'
    # )
    
    # Executa o pipeline para 'toxicidade' com 3 estados
    rodar_pipeline_vmg(
        lista_youtubers, 
        config_metrica=METRICAS_CONFIG['toxicidade'], 
        nome_analise='toxicidade' 
    )