from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

# Configuração estética global para gráficos (padrão acadêmico)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
sns.set_context("paper", font_scale=1.2)

# Define as regras para cada tipo de análise (métrica)
METRICAS_CONFIG = {
    'pysentimiento': {
        'coluna_base': 'sentimento_dominante', 
        'tipo_estados': 'categorico', 
        'estados': ['POS', 'NEU', 'NEG']
    },
    'negatividade': {
        'coluna_base': 'negatividade',
        'tipo_estados': 'numerico',
        'n_estados': 3 
    },
    'detoxify': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', 
        'limiares': [0.0, 0.20, 0.80, 1.01], 
        'estados': ['NT', 'GZ', 'T']
    },
    'perspective': {
        'coluna_base': 'p_toxicity',
        'tipo_estados': 'numerico_categorizado', 
        'limiares': [0.0, 0.20, 0.40, 1.01], 
        'estados': ['NT', 'GZ', 'T']
    }
}

console = Console()

'''
    Função para armazenar as transições de estados (para uma métrica específica) de cada vídeo em um arquivo CSV

    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'negatividade')
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
                # Cria a pasta VMG/Matrizes dentro da pasta do vídeo
                output_folder = video_path / 'VMG' / 'Matrizes'
                output_folder.mkdir(parents=True, exist_ok=True)
                
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se o DataFrame tem a coluna necessária e dados suficientes
                if df_tiras_video.empty or coluna_base not in df_tiras_video.columns or len(df_tiras_video) < 2:
                    continue

                # Lógica de Definição de Estado
                if tipo_estados == 'categorico':
                    estados = metrica_config['estados']
                    df_tiras_video.rename(columns={coluna_base: 'estado'}, inplace=True)
                    df_tiras_video = df_tiras_video[df_tiras_video['estado'].isin(estados)]

                    if len(df_tiras_video) < 2: continue

                    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)
                
                elif tipo_estados == 'numerico':
                    n = metrica_config['n_estados']
                    grupos = np.linspace(0, 1, n + 1)
                    labels = range(1, n + 1)

                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=grupos,
                        labels=labels,
                        include_lowest=True
                    )

                elif tipo_estados == 'numerico_categorizado':
                    bins = metrica_config['limiares']
                    labels = metrica_config['estados']

                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=bins,
                        labels=labels,
                        right=False, 
                        include_lowest=True 
                    )

                    tipo_categorico = CategoricalDtype(categories=labels, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)

                    if len(df_tiras_video.dropna(subset=['estado'])) < 2:
                        continue
                else:
                    console.print(f"[red]Erro: Tipo de estado '{tipo_estados}' não reconhecido.[/red]")
                    continue

                # Lógica de transição
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa e conta
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')
                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Salva no novo caminho padronizado
                output_path = output_folder / f'transicoes_{nome_analise}.csv'
                contagem.to_csv(output_path, index=False)
            
            except Exception as e:
                console.print(f'[bold red]Erro[/bold red] em {video_path.name} (salvar_transicoes_por_metrica): {e}')

'''
    Função para criar e persistir as Matrizes de Transição absolutas solicitadas para cada vídeo individual

    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'negatividade')
    @param metricas - Lista de métricas a serem calculadas e salvas
'''
def salvar_matriz_transicao_video(youtubers_list: list[str], metrica_config: dict, nome_analise: str, metricas: list[str]) -> None:
    # Filtra as métricas válidas para o nível "vídeo" (métricas que dependem de vários vídeos, como desvio padrão, são ignoradas aqui)
    metricas_absolutas_validas = [m for m in metricas if m in ['probabilidade', 'suporte', 'confianca', 'lift']]
    if not metricas_absolutas_validas:
        return

    # Definir os estados com base no tipo de métrica
    if metrica_config['tipo_estados'] in ['categorico', 'numerico_categorizado']:
        estados = metrica_config['estados']
    else:
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1))
    
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): continue

        console.print(f'>>> Processando matrizes de vídeo de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan]')

        # Procura os arquivos de transição na nova estrutura de pastas
        for transicoes_csv_path in base_path.rglob(f'VMG/Matrizes/transicoes_{nome_analise}.csv'):                
            try:                
                df_trans = pd.read_csv(transicoes_csv_path)
                if df_trans.empty: continue

                df_trans['estado'] = df_trans['estado'].astype(tipo_categorico)
                df_trans['proximo_estado'] = df_trans['proximo_estado'].astype(tipo_categorico)

                # Valores base para cálculos
                n_total = df_trans['contagem'].sum()
                n_u = df_trans.groupby('estado', observed=False)['contagem'].transform('sum')
                n_v = df_trans.groupby('proximo_estado', observed=False)['contagem'].transform('sum')

                # --- CÁLCULO DINÂMICO DAS MÉTRICAS ---
                if 'suporte' in metricas_absolutas_validas:
                    df_trans['suporte'] = (df_trans['contagem'] / n_total).fillna(0)
                
                # Cálculo idêntico para probabilidade e confiança, preservando a semântica de saída
                if 'confianca' in metricas_absolutas_validas:
                    df_trans['confianca'] = (df_trans['contagem'] / n_u).fillna(0)
                    
                if 'probabilidade' in metricas_absolutas_validas:
                    df_trans['probabilidade'] = (df_trans['contagem'] / n_u).fillna(0)
                
                if 'lift' in metricas_absolutas_validas:
                    confianca_base = (df_trans['contagem'] / n_u).fillna(0)
                    p_v = n_v / n_total
                    df_trans['lift'] = (confianca_base / p_v).fillna(0)

                # Persistir cada métrica em um CSV diferente na mesma pasta Matrizes
                output_folder = transicoes_csv_path.parent
                
                for metrica in metricas_absolutas_validas:
                    matriz = df_trans.pivot(index='estado', columns='proximo_estado', values=metrica).fillna(0)
                    output_path = output_folder / f'VMG_{nome_analise}_{metrica}.csv'
                    matriz.to_csv(output_path)

            except Exception as e:
                console.print(f'Inválido (salvar_matriz_transicao_video) em {transicoes_csv_path.parent.parent.parent.name}: {e}')

'''
    Função para criar e persistir as Matrizes de Transição (VMG) agregadas para cada youtuber

    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param metricas - Lista de métricas a serem calculadas
'''
def salvar_matriz_transicao_youtuber(youtubers_list: list[str], metrica_config: dict, nome_analise: str, metricas: list[str]) -> None:
    if metrica_config['tipo_estados'] in ['categorico', 'numerico_categorizado']:
        estados = metrica_config['estados']
    else:
        estados = list(range(1, metrica_config['n_estados'] + 1))
    
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)

    # Separa os tipos de métricas para aplicar a lógica correta
    metricas_absolutas = [m for m in metricas if m in ['probabilidade', 'suporte', 'confianca', 'lift']]
    metricas_estatisticas = [m for m in metricas if m in ['media', 'desvio_padrao', 'coeficiente_variacao']]

    # Dicionário de funções para agregação estatística
    funcoes_estatisticas = {
        'media': 'mean',
        'desvio_padrao': 'std',
        'coeficiente_variacao': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): continue

        try:
            lista_dfs = []

            # Lê a partir da estrutura de pastas (VMG/Matrizes)
            for p in base_path.rglob(f'VMG/Matrizes/transicoes_{nome_analise}.csv'):
                df = pd.read_csv(p)
                if not df.empty:
                    # Precisamos identificar o vídeo para as métricas estatísticas
                    # Ex de path: files/Youtuber/Video/VMG/Matrizes/transicoes.csv
                    df['video_id'] = p.parent.parent.parent.name
                    lista_dfs.append(df)
            
            if not lista_dfs: continue
            
            df_agg = pd.concat(lista_dfs, ignore_index=True)
            df_agg['estado'] = df_agg['estado'].astype(tipo_categorico)
            df_agg['proximo_estado'] = df_agg['proximo_estado'].astype(tipo_categorico)

            # Define a pasta de saída para o Youtuber
            output_folder = base_path / 'VMG' / 'Matrizes'
            output_folder.mkdir(parents=True, exist_ok=True)

            # BLOCO 1: MÉTRICAS ABSOLUTAS
            if metricas_absolutas:
                df_counts = df_agg.groupby(['estado', 'proximo_estado'], observed=False)['contagem'].sum().reset_index()
                n_total = df_counts['contagem'].sum()
                n_u = df_counts.groupby('estado', observed=False)['contagem'].transform('sum')
                n_v = df_counts.groupby('proximo_estado', observed=False)['contagem'].transform('sum')

                resultados_absolutos = {}
                if 'suporte' in metricas_absolutas:
                    resultados_absolutos['suporte'] = (df_counts['contagem'] / n_total).fillna(0)
                if 'confianca' in metricas_absolutas:
                    resultados_absolutos['confianca'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'probabilidade' in metricas_absolutas:
                    resultados_absolutos['probabilidade'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'lift' in metricas_absolutas:
                    confianca_base = (df_counts['contagem'] / n_u).fillna(0)
                    p_v = n_v / n_total
                    resultados_absolutos['lift'] = (confianca_base / p_v).fillna(0)

                for m_name, values in resultados_absolutos.items():
                    df_counts[m_name] = values
                    matriz = df_counts.pivot(index='estado', columns='proximo_estado', values=m_name).fillna(0)
                    matriz.to_csv(output_folder / f'VMG_{nome_analise}_{m_name}.csv')

            # BLOCO 2: MÉTRICAS ESTATÍSTICAS
            if metricas_estatisticas:
                # 1. Calcular a probabilidade base por VÍDEO
                n_u_video = df_agg.groupby(['video_id', 'estado'], observed=False)['contagem'].transform('sum')
                df_agg['prob_video'] = (df_agg['contagem'] / n_u_video).fillna(0)

                # 2. Agregar as probabilidades entre os vídeos usando a função desejada
                for m_name in metricas_estatisticas:
                    funcao = funcoes_estatisticas[m_name]
                    df_stat = df_agg.groupby(['estado', 'proximo_estado'], observed=False)['prob_video'].agg(funcao).reset_index()
                    matriz = df_stat.pivot(index='estado', columns='proximo_estado', values='prob_video').fillna(0)
                    matriz.to_csv(output_folder / f'VMG_{nome_analise}_{m_name}.csv')

        except Exception as e:
            console.print(f'[bold red]Erro[/bold red] (salvar_matriz_transicao_youtuber) para {youtuber}: {e}')

'''
    Função para criar e persistir a Matriz de Transição (VMG) global e por categoria

    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param metrica_config - Dicionário de configuração da métrica
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param granularidade - Lista com granularidades ativas (para filtrar 'categoria' ou 'geral')
    @param metricas - Lista de métricas a serem calculadas
'''
def salvar_matriz_transicao_global(mapa_categorias: dict, metrica_config: dict, nome_analise: str, granularidade: list[str], metricas: list[str]) -> None:
    if metrica_config['tipo_estados'] in ['categorico', 'numerico_categorizado']:
        estados = metrica_config['estados']
    else:
        estados = list(range(1, metrica_config['n_estados'] + 1))
    
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
    
    metricas_absolutas = [m for m in metricas if m in ['probabilidade', 'suporte', 'confianca', 'lift']]
    metricas_estatisticas = [m for m in metricas if m in ['media', 'desvio_padrao', 'coeficiente_variacao']]

    funcs_estatisticas = {
        'media': 'mean',
        'desvio_padrao': 'std',
        'coeficiente_variacao': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }

    # Coleta inicial de todos os dados
    lista_geral_transicoes = []
    for youtuber, categoria in mapa_categorias.items():
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): continue
            
        for p in base_path.rglob(f'VMG/Matrizes/transicoes_{nome_analise}.csv'):
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    df['categoria'] = categoria
                    df['youtuber'] = youtuber
                    df['video_id_unique'] = f"{youtuber}_{p.parent.parent.parent.name}"
                    lista_geral_transicoes.append(df)
            except Exception:
                continue

    if not lista_geral_transicoes:
        console.print("[yellow]Nenhum dado encontrado para processamento macro.[/yellow]")
        return

    df_master = pd.concat(lista_geral_transicoes, ignore_index=True)
    df_master['estado'] = df_master['estado'].astype(tipo_categorico)
    df_master['proximo_estado'] = df_master['proximo_estado'].astype(tipo_categorico)

    # Definir os escopos de análise com base no que foi pedido no pipeline
    escopos = []
    if 'geral' in granularidade:
        escopos.append('Geral')
    if 'categoria' in granularidade:
        # Pega as categorias únicas (ex: 'Minecraft', 'Roblox')
        escopos.extend(list(df_master['categoria'].unique()))

    for escopo in escopos:
        console.print(f'>>> Processando matrizes de "{nome_analise}" para o Escopo: [bold magenta]{escopo}[/bold magenta]')
        
        # Filtra o dataframe conforme o escopo
        if escopo == 'Geral':
            df_escopo = df_master.copy()
        else:
            df_escopo = df_master[df_master['categoria'] == escopo].copy()

        # Define a pasta de saída (Ex: files/VMG/Minecraft/Matrizes)
        output_folder = Path('files/VMG') / escopo / 'Matrizes'
        output_folder.mkdir(parents=True, exist_ok=True)

        try:
            # BLOCO 1: MÉTRICAS ABSOLUTAS
            if metricas_absolutas:
                df_counts = df_escopo.groupby(['estado', 'proximo_estado'], observed=False)['contagem'].sum().reset_index()
                n_total = df_counts['contagem'].sum()
                n_u = df_counts.groupby('estado', observed=False)['contagem'].transform('sum')
                n_v = df_counts.groupby('proximo_estado', observed=False)['contagem'].transform('sum')

                resultados_absolutos = {}
                if 'suporte' in metricas_absolutas:
                    resultados_absolutos['suporte'] = (df_counts['contagem'] / n_total).fillna(0)
                if 'confianca' in metricas_absolutas:
                    resultados_absolutos['confianca'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'probabilidade' in metricas_absolutas:
                    resultados_absolutos['probabilidade'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'lift' in metricas_absolutas:
                    confianca_base = (df_counts['contagem'] / n_u).fillna(0)
                    p_v = n_v / n_total
                    resultados_absolutos['lift'] = (confianca_base / p_v).fillna(0)

                for m_name, values in resultados_absolutos.items():
                    df_counts[m_name] = values
                    matriz = df_counts.pivot(index='estado', columns='proximo_estado', values=m_name).fillna(0)
                    matriz.to_csv(output_folder / f'VMG_{nome_analise}_{m_name}.csv')

            # BLOCO 2: MÉTRICAS ESTATÍSTICAS
            if metricas_estatisticas:
                # Calcula a probabilidade base individualmente por vídeo dentro deste escopo
                n_u_video = df_escopo.groupby(['video_id_unique', 'estado'], observed=False)['contagem'].transform('sum')
                df_escopo['prob_video'] = (df_escopo['contagem'] / n_u_video).fillna(0)

                for m_name in metricas_estatisticas:
                    func = funcs_estatisticas[m_name]
                    df_stat = df_escopo.groupby(['estado', 'proximo_estado'], observed=False)['prob_video'].agg(func).reset_index()
                    matriz = df_stat.pivot(index='estado', columns='proximo_estado', values='prob_video').fillna(0)
                    matriz.to_csv(output_folder / f'VMG_{nome_analise}_{m_name}.csv')

        except Exception as e:
            console.print(f'[red]Erro ao processar escopo {escopo}: {e}[/red]')

'''
    Função para gerar um Heatmap que mostra as métricas de transição de forma adaptativa.

    @param matrix_path - Caminho para o arquivo CSV da matriz VTMG
    @param output_path - Caminho onde a imagem será salva
    @param title - Título do gráfico
    @param metrica - Nome da métrica sendo plotada
'''
def gerar_heatmap_vmg(matrix_path: Path, output_path: Path, title: str, metrica: str):
    try:
        df_matrix = pd.read_csv(matrix_path, index_col=0)
        plt.figure(figsize=(10, 8))
        
        # Ajuste dinâmico de escala e cores dependendo da natureza da métrica
        if metrica in ['probabilidade', 'confianca', 'suporte', 'media']:
            v_min, v_max = 0.0, 1.0
            cmap = "Blues"
        elif metrica == 'lift':
            v_min, v_max = 0.0, None
            cmap = "YlOrRd"
        elif metrica == 'desvio_padrao':
            v_min, v_max = 0.0, None
            cmap = "Purples" # Roxo para dispersão absoluta
        elif metrica == 'coeficiente_variacao':
            v_min, v_max = 0.0, None
            cmap = "Oranges" # Laranja para variação relativa
        else:
            v_min, v_max = None, None
            cmap = "Greys"
        
        ax = sns.heatmap(
            df_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap, 
            linewidths=.5,
            vmin=v_min, vmax=v_max,
            cbar_kws={'label': metrica.replace('_', ' ').title()}
        )
        
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel("Estado de Destino (V)", fontsize=12)
        ax.set_ylabel("Estado de Origem (U)", fontsize=12)
        
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        console.print(f"[red]Erro no heatmap ({metrica}): {e}[/red]")

'''
    Função que gera os gráficos individuais para cada matriz de vídeo encontrada

    @param youtubers_list - Lista de youtubers a serem analisados
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param metricas - Lista de métricas a serem plotadas
'''
def gerar_visualizacoes_video(youtubers_list: list[str], nome_analise: str, metricas: list[str]):
    console.print(f"\n[bold magenta]===== GERANDO HEATMAPS DE VÍDEO ({nome_analise.upper()}) =====[/bold magenta]")
    
    # Filtra apenas as métricas absolutas, que são as únicas geradas no nível de vídeo
    metricas_video = [m for m in metricas if m in ['probabilidade', 'suporte', 'confianca', 'lift']]
    
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): continue
        
        for metrica in metricas_video:
            arquivos_vmg = list(base_path.rglob(f'VMG/Matrizes/VMG_{nome_analise}_{metrica}.csv'))
            if not arquivos_vmg: continue
            
            for vmg_file in arquivos_vmg:
                nome_video = vmg_file.parent.parent.parent.name
                
                # Redireciona a saída para a pasta Plots
                pasta_plot = vmg_file.parent.parent / 'Plots'
                pasta_plot.mkdir(exist_ok=True)
                
                output_file = pasta_plot / f'heatmap_{nome_analise}_{metrica}.png'
                titulo_grafico = f"VTMG ({nome_analise}) ({metrica.replace('_', ' ').title()}): {nome_video}"
                
                gerar_heatmap_vmg(vmg_file, output_file, titulo_grafico, metrica)

'''
    Função para gerar o Heatmap das matrizes agregadas do Youtuber

    @param youtubers_list - Lista de youtubers a serem analisados
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param metricas - Lista de métricas a serem plotadas
'''
def gerar_visualizacoes_youtuber(youtubers_list: list[str], nome_analise: str, metricas: list[str]):
    console.print(f"\n[bold magenta]===== GERANDO HEATMAPS POR YOUTUBER ({nome_analise.upper()}) =====[/bold magenta]")
    
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}/VMG/Matrizes')
        img_output_folder = Path(f'files/{youtuber}/VMG/Plots')
        img_output_folder.mkdir(parents=True, exist_ok=True)
        
        for metrica in metricas:
            matriz_path = base_path / f'VMG_{nome_analise}_{metrica}.csv'
            
            if matriz_path.exists():
                heatmap_path = img_output_folder / f'heatmap_{nome_analise}_{metrica}.png'
                titulo = f"VTMG ({nome_analise}) Youtuber - {metrica.replace('_', ' ').title()}: {youtuber}"
                
                gerar_heatmap_vmg(matriz_path, heatmap_path, titulo, metrica)

'''
    Função para gerar Heatmaps das Matrizes de Transição (VMG) globais e por categoria

    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param granularidade - Lista com granularidades ativas
    @param metricas - Lista de métricas a serem plotadas
'''
def gerar_visualizacoes_globais(mapa_categorias: dict, nome_analise: str, granularidade: list[str], metricas: list[str]):
    console.print(f"\n[bold magenta]===== GERANDO HEATMAPS MACRO ({nome_analise.upper()}) =====[/bold magenta]")
    
    # Define os escopos com base no pipeline
    escopos = []
    if 'geral' in granularidade:
        escopos.append('Geral')
    if 'categoria' in granularidade:
        escopos.extend(list(set(mapa_categorias.values())))
    
    for escopo in escopos:
        base_path = Path(f'files/VMG/{escopo}/Matrizes')
        img_output_folder = Path(f'files/VMG/{escopo}/Plots')
        img_output_folder.mkdir(parents=True, exist_ok=True)
        
        for metrica in metricas:
            matriz_path = base_path / f'VMG_{nome_analise}_{metrica}.csv'
            
            if matriz_path.exists():
                heatmap_path = img_output_folder / f'heatmap_{nome_analise}_{metrica}.png'
                
                titulo_escopo = "Geral" if escopo == 'Geral' else escopo
                titulo = f"VTMG ({nome_analise}) {titulo_escopo} - {metrica.replace('_', ' ').title()}"
                
                try:
                    gerar_heatmap_vmg(matriz_path, heatmap_path, titulo, metrica)
                    console.print(f"   [green]Plot salvo ({metrica}):[/green] {escopo}")
                except Exception as e:
                    console.print(f"   [red]Erro no plot {escopo} ({metrica}): {e}[/red]")

'''
    Função principal para orquestrar o pipeline de análise de VMG parametrizado.

    @param youtubers_list - Lista de youtubers a serem analisados
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param config_metrica - Dicionário de configurações da métrica (METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'pysentimiento', 'detoxify')
    @param granularidade - Nível(is) de agrupamento ('video', 'youtuber', 'categoria', 'geral' ou lista)
    @param metricas - Tipo(s) de métrica a ser calculada (ex: 'probabilidade', 'media', 'lift', etc. ou lista)
'''
def rodar_pipeline_vmg(
    youtubers_list: list[str], 
    mapa_categorias: dict, 
    config_metrica: dict, 
    nome_analise: str,
    granularidade: list[str] | str,
    metricas: list[str] | str
):
    # Garantir que os parâmetros de controle sejam sempre listas para facilitar a iteração
    if isinstance(granularidade, str): granularidade = [granularidade]
    if isinstance(metricas, str): metricas = [metricas]

    console.print(f"\n[bold magenta]===== PIPELINE VMG: {nome_analise.upper()} =====[/bold magenta]")
    console.print(f"Granularidade(s) selecionada(s): {granularidade}")
    console.print(f"Métrica(s) selecionada(s): {metricas}\n")

    # 1. Passo Base (Sempre Necessário): Gerar as contagens absolutas de transição por vídeo
    # Como todos os níveis dependem disso, essa função roda independente da granularidade escolhida
    salvar_transicoes_por_metrica(youtubers_list, config_metrica, nome_analise)
    
    # 2. Roteamento por Nível de Granularidade
    if 'video' in granularidade:
        console.print(f"[bold blue]--- Processando Nível: VÍDEO ---[/bold blue]")
        salvar_matriz_transicao_video(youtubers_list, config_metrica, nome_analise, metricas)
        # gerar_visualizacoes_video(youtubers_list, nome_analise, metricas)

    if 'youtuber' in granularidade:
        console.print(f"\n[bold blue]--- Processando Nível: YOUTUBER ---[/bold blue]")
        salvar_matriz_transicao_youtuber(youtubers_list, config_metrica, nome_analise, metricas)
        gerar_visualizacoes_youtuber(youtubers_list, nome_analise, metricas)

    if 'categoria' in granularidade or 'geral' in granularidade:
        console.print(f"\n[bold blue]--- Processando Níveis: CATEGORIA / GERAL ---[/bold blue]")
        # Passamos as granularidades ativas e as métricas para a função global
        salvar_matriz_transicao_global(mapa_categorias, config_metrica, nome_analise, granularidade, metricas)
        gerar_visualizacoes_globais(mapa_categorias, nome_analise, granularidade, metricas)

    console.print(f"\n[bold green]✓ Pipeline {nome_analise.upper()} finalizado com sucesso![/bold green]")

if __name__ == '__main__':
    # Mapeia cada youtuber para sua categoria principal (Minecraft, Roblox, etc.)
    mapa_youtubers_categoria = {
        'Julia MineGirl': 'Roblox',
        'Papile': 'Roblox',
        'Tex HS': 'Roblox',
        'Amy Scarlet': 'Roblox',
        'Luluca Games': 'Roblox',
        'meu nome é david': 'Roblox',
        'Lokis': 'Roblox',

        'Robin Hood Gamer': 'Minecraft',
        'AuthenticGames': 'Minecraft',
        'Cadres': 'Minecraft',
        'Athos': 'Minecraft',
        'JP Plays': 'Minecraft',
        'Marcelodrv': 'Minecraft',
        'Geleia': 'Minecraft',
        'Kass e KR': 'Minecraft',
    }

    lista_youtubers = list(mapa_youtubers_categoria.keys())

    # Executa o pipeline para a métrica 'pysentimiento'
    # rodar_pipeline_vmg(
    #     youtubers_list=lista_youtubers,
    #     mapa_categorias=mapa_youtubers_categoria,
    #     config_metrica=METRICAS_CONFIG['pysentimiento'],
    #     nome_analise='pysentimiento',
    #     granularidade=['youtuber', 'categoria', 'geral'],
    #     metricas=['media', 'coeficiente_variacao', 'lift']
    # )
    
    # # Executa o pipeline para a métrica 'negatividade' com 3 estados
    # rodar_pipeline_vmg(
    #     youtubers_list=lista_youtubers,
    #     mapa_categorias=mapa_youtubers_categoria,
    #     config_metrica=METRICAS_CONFIG['negatividade'],
    #     nome_analise='negatividade',
    #     granularidade=['youtuber', 'categoria', 'geral'],
    #     metricas=['media', 'coeficiente_variacao', 'lift']
    # )
    
    # Executa o pipeline para 'toxicidade' com 3 estados
    rodar_pipeline_vmg(
        youtubers_list=lista_youtubers,
        mapa_categorias=mapa_youtubers_categoria,
        config_metrica=METRICAS_CONFIG['detoxify'],
        nome_analise='detoxify',
        granularidade=['video', 'youtuber', 'categoria', 'geral'],
        metricas=['probabilidade', 'media', 'desvio_padrao', 'coeficiente_variacao']
    )

    rodar_pipeline_vmg(
        youtubers_list=lista_youtubers,
        mapa_categorias=mapa_youtubers_categoria,
        config_metrica=METRICAS_CONFIG['perspective'],
        nome_analise='perspective',
        granularidade=['video', 'youtuber', 'categoria', 'geral'],
        metricas=['probabilidade', 'media', 'desvio_padrao', 'coeficiente_variacao']
    )