import pandas as pd
from pathlib import Path
from rich.console import Console
from prefixspan import PrefixSpan
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

# Configuração global
BASE_DATA_FOLDER = Path('files/sequencias')   

# Cores oficiais para os nós do Sankey
CORES_ESTADOS = {
    'NT': '#2ecc71',   # Verde
    'GZ': '#f1c40f',   # Amarelo/Ouro
    'T': '#e74c3c',    # Vermelho
    'EVENTO': '#34495e' # Cinza Escuro (O Gatilho)
}

# Configuração estética global (Estilo Acadêmico Clean)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid", context="paper")

'''
    Prepara listas específicas para o algoritmo de mineração (PrefixSpan/Apriori).
    Permite escolher se queremos olhar para o passado, futuro ou tudo.

    @param df - O DataFrame carregado
    @param modo - 'full' (tudo), 'pre' (antes do evento), 'pos' (depois do evento)
    @return list[list[str]] - Lista de listas pronta para o algoritmo
'''
def carregar_listas_para_mineracao(caminho_arquivo: str, modo: str = 'full') -> list[list[str]]:
    try:
        df = pd.read_csv(caminho_arquivo)

        # Identifica a coluna do evento
        idx_evento = len(df.columns) // 2
        try:
            idx_evento = df.columns.get_loc('evento')
        except KeyError:
            # Fallback se não achar a coluna 'evento': pega o meio
            idx_evento = len(df.columns) // 2

        # Remove o video_id para a mineração de padrões
        colunas_dados = [c for c in df.columns if c != 'video_id']
        
        # Recalcula o índice do evento baseado apenas nas colunas de dados
        colunas_pre = [c for c in colunas_dados if c.startswith('t-')]
        colunas_pos = [c for c in colunas_dados if c.startswith('t+')]
        
        listas_finais = []

        if modo == 'pre':
            # Pega de t-N até t-1
            # Filtra o df apenas com colunas que começam com 't-'
            df_recorte = df[colunas_pre]
            listas_finais = df_recorte.astype(str).values.tolist()

        elif modo == 'pos':
            # Pega de t+1 até t+N
            df_recorte = df[colunas_pos]
            listas_finais = df_recorte.astype(str).values.tolist()

        else: # 'full'
            # Pega tudo (menos o ID)
            # Útil para ver pontes, ex: "GZ -> EVENTO -> GZ"
            df_recorte = df[colunas_dados]
            listas_finais = df_recorte.astype(str).values.tolist()
            
        return listas_finais

    except Exception as e:
        console.print(f"     [red]Erro ao carregar sequências de {caminho_arquivo.name}: {e}[/red]")
        return pd.DataFrame()

'''
    Função que aplica o algoritmo PrefixSpan
    Inclui filtros de tamanho e retorna dados estruturados com percentuais

    @param database_sequencias - A lista de listas (recorte pré ou pós evento)
    @param min_suporte_percent - A porcentagem mínima (0 a 100)
    @param min_tamanho - Ignora padrões menores que isso (padrão: 1)
    @param max_tamanho - Ignora padrões maiores que isso (padrão: 5)
    
    @return list[dict] - Lista de dicionários ordenados por relevância
'''
def minerar_padroes_frequentes(
    database_sequencias: list[list[str]], 
    min_suporte_percent: float,
    min_tamanho: int = 1,
    max_tamanho: int = 10
) -> list[dict]:
    
    total_sequencias = len(database_sequencias)
    if total_sequencias == 0:
        return []
    
    # Calcula o suporte absoluto
    min_suporte_absoluto = int(total_sequencias * (min_suporte_percent / 100))

    # Garante que seja pelo menos 1
    min_suporte_absoluto = max(1, min_suporte_absoluto)

    console.print(f"     Minerando {total_sequencias} sequências (Min Suporte: {min_suporte_absoluto} | {min_suporte_percent}%)")

    # Executa o PrefixSpan
    ps = PrefixSpan(database_sequencias)
    
    # Encontra os padrões frequentes
    resultados_brutos = ps.frequent(min_suporte_absoluto)
    
    resultados_estruturados = []

    # Processa e filtra os resultados
    for count, padrao in resultados_brutos:
        tamanho_padrao = len(padrao)
        
        # Filtra pelo tamanho desejado
        if tamanho_padrao < min_tamanho or tamanho_padrao > max_tamanho:
            continue
            
        percentual = (count / total_sequencias) * 100
        
        # Cria um objeto rico para análise posterior
        resultados_estruturados.append({
            'padrao': padrao,                 # Lista ['NT', 'T']
            'padrao_str': " -> ".join(padrao), # String "NT -> T"
            'suporte_abs': count,
            'suporte_rel': percentual,
            'tamanho': tamanho_padrao
        })

    # Ordena: Primeiro por Suporte (Decrescente), depois por Tamanho (Decrescente)
    resultados_estruturados.sort(key=lambda x: (x['suporte_abs'], x['tamanho']), reverse=True)
    
    return resultados_estruturados

'''
    Função auxiliar para determinar a cor da barra baseada na semântica do padrão
    Vermelho = Contém T (Perigo)
    Amarelo = Contém GZ mas não T (Alerta)
    Verde = Apenas NT (Seguro)
'''
def obter_cor_semantica(padrao_str: str) -> str:
    # Cores
    COR_PERIGO = "#e74c3c" # Vermelho
    COR_ALERTA = "#f1c40f" # Amarelo/Ouro
    COR_SEGURO = "#2ecc71" # Verde
    COR_NEUTRO = "#95a5a6" # Cinza (caso de fallback)

    # Lógica de prioridade: O pior estado define a cor
    if 'T' in padrao_str.split(' -> '): # Split para evitar que 'NT' triggue o 'T'
        return COR_PERIGO
    elif 'GZ' in padrao_str:
        return COR_ALERTA
    elif 'NT' in padrao_str:
        return COR_SEGURO
    else:
        return COR_NEUTRO

'''
    Gera um gráfico de barras horizontais dos Top N padrões mais frequentes.
    Salva como imagem de alta resolução pronta para slides/artigos.
'''
def gerar_grafico_barras_padroes(df: pd.DataFrame, caminho_saida: Path, top_n: int = 15):
    if df.empty: return

    # Filtra os Top N
    df_plot = df.head(top_n).copy()
    
    # Cria a coluna de cores baseada na função semântica
    df_plot['cor'] = df_plot['Padrao'].apply(obter_cor_semantica)

    # Configura a figura
    altura_fig = max(6, top_n * 0.4)
    fig, ax = plt.subplots(figsize=(10, altura_fig))

    # Desenha as barras
    bars = ax.barh(
        y=df_plot['Padrao'], 
        width=df_plot['Suporte Percentual'], 
        color=df_plot['cor'],
        edgecolor='none',
        height=0.6,
        alpha=0.9
    )

    # Inverte o eixo Y para o Top 1 ficar em cima
    ax.invert_yaxis()

    # Títulos e Eixos
    ax.set_title("Padrões Sequenciais Mais Frequentes", fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel("Suporte (%) - Frequência de Ocorrência", fontsize=11, fontweight='bold')
    ax.set_ylabel("Padrão Sequencial", fontsize=11, fontweight='bold')
    
    # Remove bordas desnecessárias (clean look)
    sns.despine(left=True, bottom=True)
    
    # Adiciona grid vertical suave
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(False)

    # Anotação de valores (Data Labels)
    # Pesquisadores exigentes querem ver o número exato, não apenas a barra
    for bar in bars:
        width = bar.get_width()
        label_y = bar.get_y() + bar.get_height() / 2
        
        # Posição do texto um pouco à direita da barra
        ax.text(
            width + 0.5, # Offset X
            label_y,     # Posição Y (centro da barra)
            f'{width:.1f}%', 
            va='center', 
            ha='left', 
            fontsize=10, 
            color='#333333',
            fontweight='bold'
        )

    # Ajuste de margens para o texto não cortar
    # Aumenta o limite direito do gráfico em 10% para caber os números
    ax.set_xlim(0, df_plot['Suporte Percentual'].max() * 1.15)
    
    # Aumenta a fonte dos labels do eixo Y para legibilidade
    ax.tick_params(axis='y', labelsize=11)

    try:
        plt.tight_layout()
        plt.savefig(caminho_saida, bbox_inches='tight')
        plt.close()
        console.print(f"     [green]Gráfico salvo:[/green] {caminho_saida}")
    except Exception as e:
        console.print(f"     [red]Erro ao salvar gráfico: {e}[/red]")

'''
    Função para formatar os resultados minerados e salvar em um CSV legível.
    Adapta-se ao novo formato de retorno da função de mineração (lista de dicionários).

    @param resultados_minerados - A lista de dicionários retornada por minerar_padroes_frequentes
    @param caminho_saida - O Path onde o relatório final será salvo
'''
def formatar_e_salvar_resultados(resultados_minerados: list[dict], caminho_saida: Path) -> None:
    if not resultados_minerados:
        console.print(f"     [yellow]Nenhum padrão frequente encontrado para salvar em {caminho_saida.name}.[/yellow]")
        return

    # Prepara dados para o DataFrame
    dados_csv = []
    
    for item in resultados_minerados:
        dados_csv.append({
            'Padrao': item['padrao_str'],
            'Tamanho': item['tamanho'],
            'Suporte Absoluto': item['suporte_abs'],
            'Suporte Percentual': round(item['suporte_rel'], 2)
        })
    
    # Cria DataFrame
    df_resultados = pd.DataFrame(dados_csv)
    
    # Salva em CSV
    try:
        df_resultados.to_csv(caminho_saida, index=False)
        console.print(f"[green]Resultados salvos:[/green] {caminho_saida}")
    except Exception as e:
        console.print(f"[red]Erro ao escrever arquivo CSV: {e}[/red]")
        return
    
    # 3. Gera Gráfico de Barras (Top 15)
    caminho_imagem = caminho_saida.parent / f"grafico_{caminho_saida.stem}.png"
    gerar_grafico_barras_padroes(df_resultados, caminho_imagem, top_n=15)

    # Mostra prévia dos top 10 padrões no console
    console.print(f"[bold]Top Padrões Encontrados:[/bold]")
    
    # Pegamos os 10 primeiros
    limit = min(10, len(dados_csv))
    
    for i in range(limit):
        row = dados_csv[i]
        padrao_display = row['Padrao']
        perc_display = row['Suporte Percentual']
        
        # Formatação condicional simples para o console
        cor = "white"
        
        console.print(f"     #{i+1:02d}: [{cor}]{padrao_display}[/{cor}] (Suporte: {perc_display}%)")

'''
    Gera um Diagrama de Sankey (Plotly) visualizando os fluxos dos padrões mais frequentes.
    
    @param resultados - Lista de dicionários com os padrões minerados
    @param modo - 'pre' (Causa) ou 'pos' (Consequência)
    @param output_html - Caminho para salvar o HTML interativo
    @param titulo - Título do gráfico
'''
def gerar_sankey_sequencias(resultados: list[dict], modo: str, output_html: Path, titulo: str):
    if not resultados:
        return

    # Preparação dos nós e links    
    node_labels = [] # Lista de nomes para exibir (NT, T, GZ)
    node_colors = [] # Cores dos nós
    node_map = {}    # Mapa (step, label) -> index na lista node_labels
    
    sources = []
    targets = []
    values = []
    
    # Adiciona o nó central "EVENTO"
    idx_evento = 0
    node_labels.append("EVENTO (Gatilho)")
    node_colors.append(CORES_ESTADOS['EVENTO'])
    node_map[('evento', 'EVENTO')] = idx_evento
    
    # Pega apenas os Top 15 padrões para não poluir o gráfico
    top_patterns = resultados[:15]
    
    for item in top_patterns:
        padrao = item['padrao'] # Ex: ['NT', 'GZ']
        peso = item['suporte_abs']
        
        last_node_idx = -1
        
        # Lógica para CAUSA (Pre): O fluxo vai DO padrão PARA o evento
        # Ex: padrao=['NT', 'GZ']. Fluxo: NT(t-2) -> GZ(t-1) -> EVENTO
        if modo == 'pre':
            # Começa do início do padrão
            for i, estado in enumerate(padrao):
                # O passo temporal é relativo ao fim da lista
                # Se len=2: 'NT' é step -2, 'GZ' é step -1
                step = i - len(padrao) 
                
                key = (step, estado)
                
                # Se nó não existe, cria
                if key not in node_map:
                    node_map[key] = len(node_labels)
                    node_labels.append(estado)
                    node_colors.append(CORES_ESTADOS.get(estado, '#cccccc'))
                
                curr_idx = node_map[key]
                
                # Se não é o primeiro nó da corrente, conecta com o anterior
                if i > 0:
                    sources.append(last_node_idx)
                    targets.append(curr_idx)
                    values.append(peso)
                
                last_node_idx = curr_idx
            
            # Conecta o último nó do padrão ao Evento
            sources.append(last_node_idx)
            targets.append(idx_evento)
            values.append(peso)

        # Lógica para CONSEQUÊNCIA (Pos): O fluxo vai DO evento PARA o padrão
        # Ex: padrao=['T', 'NT']. Fluxo: EVENTO -> T(t+1) -> NT(t+2)
        elif modo == 'pos':
            last_node_idx = idx_evento
            
            for i, estado in enumerate(padrao):
                step = i + 1 # t+1, t+2...
                key = (step, estado)
                
                if key not in node_map:
                    node_map[key] = len(node_labels)
                    node_labels.append(estado)
                    node_colors.append(CORES_ESTADOS.get(estado, '#cccccc'))
                
                curr_idx = node_map[key]
                
                sources.append(last_node_idx)
                targets.append(curr_idx)
                values.append(peso)
                
                last_node_idx = curr_idx

    # Criação do Gráfico Plotly
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = node_colors
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values
        )
    )])

    fig.update_layout(title_text=titulo, font_size=12)
    
    # Salva
    try:
        fig.write_html(output_html)
        console.print(f"     [green]Sankey salvo:[/green] {output_html}")
    except Exception as e:
        console.print(f"     [red]Erro ao salvar Sankey: {e}[/red]")

'''
    Função principal para orquestrar a mineração em todos os arquivos preparados

    @param tipo_analise - 'toxicidade', 'negatividade' ou 'misto_9_estados'
    @param min_suporte - Porcentagem mínima de frequência (padrão 5%)
'''
def orquestrar_mineracao_sequencias(tipo_analise: str, min_suporte: float = 5.0):
    console.print(f"\n[bold magenta]=== Iniciando Mineração de Sequências ({tipo_analise.upper()}) ===[/bold magenta]")
    console.print(f"Buscando padrões com suporte mínimo de: {min_suporte}%")

    pasta_busca = BASE_DATA_FOLDER 
    
    if not pasta_busca.is_dir():
        console.print(f"[red]Pasta de sequências não encontrada: {pasta_busca}[/red]")
        return

    # Busca todos os arquivos CSV recursivamente que correspondem ao tipo de análise
    padrao_busca = f"sequencias_{tipo_analise}_*.csv"
    arquivos_dataset = list(pasta_busca.rglob(padrao_busca))

    if not arquivos_dataset:
        console.print(f"[yellow]Nenhum dataset encontrado com o padrão: {padrao_busca}[/yellow]")
        return

    # Itera por cada arquivo encontrado (cada grupo/youtuber)
    for caminho_dataset in arquivos_dataset:
        nome_grupo = caminho_dataset.parent.name
        console.print(f"\n[bold cyan]--------------------------------------------------[/bold cyan]")
        console.print(f"[bold cyan]Processando Grupo: {nome_grupo}[/bold cyan]")
        console.print(f"[dim]Arquivo: {caminho_dataset.name}[/dim]")
        
        # Cria pasta para plots se não existir
        pasta_plots = caminho_dataset.parent / 'plots'
        pasta_plots.mkdir(exist_ok=True)

        # --- Analisando Causas (Pré-Evento) ---
        console.print(f"\n[bold]A. Analisando Causas (Pré-Evento)[/bold]")
        seqs_pre = carregar_listas_para_mineracao(caminho_dataset, modo='pre')

        resultados_pre = minerar_padroes_frequentes(
            seqs_pre, 
            min_suporte_percent=min_suporte, 
            min_tamanho=2, # Padrões com pelo menos 2 passos para ter fluxo
            max_tamanho=5
        )
        
        # Salva Relatório PRE
        nome_saida_pre = f"padroes_CAUSA_{caminho_dataset.stem}.csv"
        formatar_e_salvar_resultados(resultados_pre, caminho_dataset.parent / nome_saida_pre)
        
        # Gera Sankey PRE
        sankey_pre_path = pasta_plots / f"sankey_CAUSA_{caminho_dataset.stem}.html"
        gerar_sankey_sequencias(
            resultados_pre, 
            'pre', 
            sankey_pre_path, 
            f"Fluxo de Causa (O que leva à Toxicidade?) - {nome_grupo}"
        )

        # --- Analisando Consequências (Pós-Evento) ---
        console.print(f"\n[bold]B. Analisando Consequências (Pós-Evento)[/bold]")
        seqs_pos = carregar_listas_para_mineracao(caminho_dataset, modo='pos')
        
        resultados_pos = minerar_padroes_frequentes(
            seqs_pos, 
            min_suporte_percent=min_suporte, 
            min_tamanho=2, # Padrões com pelo menos 2 passos para ter fluxo
            max_tamanho=5
        )

        # Salva Relatório POS
        nome_saida_pos = f"padroes_CONSEQUENCIA_{caminho_dataset.stem}.csv"
        formatar_e_salvar_resultados(resultados_pos, caminho_dataset.parent / nome_saida_pos)
        
        # Gera Sankey POS
        sankey_pos_path = pasta_plots / f"sankey_CONSEQUENCIA_{caminho_dataset.stem}.html"
        gerar_sankey_sequencias(
            resultados_pos, 
            'pos', 
            sankey_pos_path, 
            f"Fluxo de Consequência (O que acontece após?) - {nome_grupo}"
        )

    console.print("\n[bold green]=== Processamento Concluído ===[/bold green]")

if __name__ == "__main__":
    MIN_SUPORTE_PERCENT = 10.0 
    
    orquestrar_mineracao_sequencias('toxicidade', MIN_SUPORTE_PERCENT)    
    #orquestrar_mineracao_sequencias('misto_9_estados', MIN_SUPORTE_PERCENT)    
    # orquestrar_mineracao_sequencias('sentimento', MIN_SUPORTE_PERCENT)