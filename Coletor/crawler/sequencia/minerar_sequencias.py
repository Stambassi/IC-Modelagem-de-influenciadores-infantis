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
    Função para calcular a probabilidade absoluta de ocorrência de cada estado (NT, GZ, T) no banco de dados

    @param database_sequencias - A lista de sequências brutas
    @return dict - Dicionário com a frequência relativa de cada estado
'''
def calcular_probabilidades_basais(database_sequencias: list[list[str]]) -> dict:
    contagem = {'NT': 0, 'GZ': 0, 'T': 0}
    total_elementos = 0
    
    # Contar o número de ocorrências de cada estado
    for seq in database_sequencias:
        for estado in seq:
            if estado in contagem:
                contagem[estado] += 1
                total_elementos += 1
                
    # Se não forem encontrados elementos, retorna probabilidade 0
    if total_elementos == 0: return {k: 0 for k in contagem}
    
    # Para cada estado (NT, GZ e T), calcula a probabilidade de ocorrência dividindo
    # o número de ocorrências pelo total de elementos
    return {k: v / total_elementos for k, v in contagem.items()}

'''
    Função para calcular o Lift e Confiança para padrões em relação a um estado alvo
    O Lift indica o quanto a presença do padrão altera a probabilidade do evento ocorrer

    @param resultados_minerados - Lista de dicionários com os padrões encontrados
    @param database_controle - Sequências brutas para cálculo da probabilidade basal
    @param modo - 'pre' (causas), 'pos' (consequências) ou 'full' (análise geral)
    @param estado_alvo - O estado de interesse para o cálculo (ex: 'T')
    
    @return list[dict] - Resultados com as métricas de relevância estatística
'''
def calcular_relevancia_lift(resultados_minerados: list[dict], database_controle: list[list[str]], modo: str = 'pre', estado_alvo: str = 'T') -> list[dict]: 
    # Cacular a probabilidade basal do estado alvo (P(Alvo)) no conjunto de controle
    probs_basais = calcular_probabilidades_basais(database_controle)
    p_alvo = probs_basais.get(estado_alvo, 0.0)
    
    # Teste para evitar divisão por zero se o estado alvo não existir no controle
    if p_alvo == 0:
        p_alvo = 0.0001 
    
    for item in resultados_minerados:
        # No seu pipeline, o suporte_rel representa a frequência do padrão 
        # dentro de janelas que JÁ são filtradas pela ocorrência do evento.
        
        # Confiança: P(Alvo | Padrão)
        # Em 'pre' ou 'pos', as sequências foram extraídas justamente por estarem
        # vinculadas ao evento, logo o suporte relativo é a nossa confiança observada.
        confianca = item['suporte_rel'] / 100 
        
        # Lift: Razão entre a probabilidade condicionada e a probabilidade basal
        # Lift = P(Alvo | Padrão) / P(Alvo)
        lift = confianca / p_alvo
        
        item['confianca'] = round(confianca, 4)
        item['lift'] = round(lift, 4)
        item['modo_analise'] = modo
        
    # Ordenação por Lift (Decrescente) para evidenciar os padrões de maior relevância
    resultados_minerados.sort(key=lambda x: x['lift'], reverse=True)
    
    return resultados_minerados

'''
    Função para preparar listas específicas para o algoritmo de mineração
    Diferencia entre arquivos 'pre' (sequência -> alvo) e 'pos' (origem -> sequência)

    @param caminho_arquivo - Path para o arquivo CSV global
    @param modo - 'pre' ou 'pos'
    @return tuple - (lista_de_listas_de_estados, lista_de_rotulos_evento)
'''
def carregar_listas_para_mineracao(caminho_arquivo: str, modo: str = 'pre') -> tuple:
    try:
        df = pd.read_csv(caminho_arquivo)
        
        # Filtra colunas que representam os passos temporais (t-n ou t+n)
        prefixo = 't-' if modo == 'pre' else 't+'
        colunas_sequencia = [c for c in df.columns if c.startswith(prefixo)]
        
        # Extrai as sequências como listas de strings
        listas_estados = df[colunas_sequencia].astype(str).values.tolist()
        
        # Extrai os rótulos que indicam se aquele momento foi um evento (SIM/NAO)
        # Fundamental para o cálculo de Confiança e Lift
        rotulos_evento = df['foi_evento'].tolist()
        
        return listas_estados, rotulos_evento

    except Exception as e:
        console.print(f"     [red]Erro ao carregar sequências globais: {e}[/red]")
        return [], []

'''
    Função que aplica o algoritmo PrefixSpan e calcula métricas de relevância (Suporte, Confiança e Lift)

    @param database_sequencias - A lista de listas de estados
    @param rotulos_evento - Lista SIM/NAO correspondente a cada sequência
    @param min_suporte_percent - Porcentagem mínima (0 a 100)
    @param modo - 'pre' ou 'pos'
    @param min_tamanho - Tamanho mínimo do padrão
    
    @return list[dict] - Lista de dicionários com Suporte, Confiança e Lift
'''
def minerar_padroes_frequentes(
    database_sequencias: list[list[str]], 
    rotulos_evento: list[str],
    min_suporte_percent: float,
    modo: str = 'pre',
    min_tamanho: int = 1,
    max_tamanho: int = 10
) -> list[dict]:
    
    total_instancias = len(database_sequencias)
    if total_instancias == 0: return []
    
    # Filtra apenas as sequências que resultaram em evento (SIM)
    # para a mineração do PrefixSpan (foco no que causa/sucede a toxicidade)
    sequencias_evento = [s for s, r in zip(database_sequencias, rotulos_evento) if r == 'SIM']
    total_eventos = len(sequencias_evento)
    
    if total_eventos == 0: return []

    min_suporte_abs = max(1, int(total_eventos * (min_suporte_percent / 100)))
    console.print(f"     Minerando {total_eventos} sequências de evento (Suporte Abs: {min_suporte_abs})")

    ps = PrefixSpan(sequencias_evento)
    resultados_brutos = ps.frequent(min_suporte_abs)
    
    resultados_estruturados = []
    
    # Cálculo da Probabilidade Basal do Evento no dataset global P(E)
    prob_basal_evento = rotulos_evento.count('SIM') / total_instancias

    for count, padrao in resultados_brutos:
        if len(padrao) < min_tamanho or len(padrao) > max_tamanho:
            continue
            
        # Suporte Relativo ao subconjunto de eventos
        suporte_no_evento = (count / total_eventos)
        
        # Cálculo da Confiança: P(Evento | Padrão)
        # É preciso contar quantas vezes o padrão aparece no dataset INTEIRO
        count_global = 0
        for seq in database_sequencias:
            # Checagem simples de subsequência (ordem preservada)
            iterator = iter(seq)
            if all(item in iterator for item in padrao):
                count_global += 1
        
        # Confiança = Ocorrências com evento / Total de ocorrências do padrão
        count_com_evento = count
        confianca = count_com_evento / count_global if count_global > 0 else 0
        
        # Lift = Confiança / Probabilidade Basal
        lift = confianca / prob_basal_evento if prob_basal_evento > 0 else 0
        
        resultados_estruturados.append({
            'padrao': padrao,
            'padrao_str': " -> ".join(padrao),
            'suporte_abs': count,
            'suporte_rel': suporte_no_evento * 100,
            'confianca': round(confianca, 4),
            'lift': round(lift, 4),
            'tamanho': len(padrao)
        })

    # Ordena por Lift para priorizar padrões com maior significância estatística
    resultados_estruturados.sort(key=lambda x: x['lift'], reverse=True)
    
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
    Função para gerar um gráfico de barras horizontais dos Top N padrões, parametrizado por métrica
    Ajusta automaticamente ordenação, títulos, cores e labels com base na métrica escolhida

    @param df - O DataFrame com os resultados da mineração (deve conter as colunas da métrica)
    @param caminho_base - O Path base para salvar a imagem (o sufixo da métrica será adicionado)
    @param tipo_sequencia - 'precursora' ou 'sucessora'
    @param metrica - 'suporte', 'confianca' ou 'lift'
    @param top_n - Quantidade de padrões a exibir
'''
def gerar_grafico_barras_padroes(df: pd.DataFrame, caminho_base: Path, tipo_sequencia: str, metrica: str = 'suporte', top_n: int = 15):
    if df.empty: return

    # Mapeamento de configurações por métrica
    config_metrica = {
        'suporte': {
            'coluna': 'Suporte Percentual',
            'label_eixo': 'Suporte (%) - Frequência de Ocorrência',
            'titulo_sufixo': 'por Suporte',
            'formato_label': '{:.1f}%',
            'sufixo_arquivo': 'suporte'
        },
        'confianca': {
            'coluna': 'Confianca',
            'label_eixo': 'Confiança - $P(Evento \\mid Padrão)$',
            'titulo_sufixo': 'por Confiança',
            'formato_label': '{:.4f}',
            'sufixo_arquivo': 'confianca'
        },
        'lift': {
            'coluna': 'Lift',
            'label_eixo': 'Lift - Significância Estatística',
            'titulo_sufixo': 'por Lift',
            'formato_label': '{:.2f}',
            'sufixo_arquivo': 'lift'
        }
    }

    if metrica not in config_metrica:
        console.print(f"[red]Métrica '{metrica}' inválida para o gráfico.[/red]")
        return

    conf = config_metrica[metrica]
    col_alvo = conf['coluna']

    # Ordenação e Filtro: Garante que o gráfico mostre os Top N da métrica escolhida
    df_plot = df.sort_values(by=col_alvo, ascending=False).head(top_n).copy()
    
    # Cria a coluna de cores baseada na função semântica
    df_plot['cor'] = df_plot['Padrao'].apply(obter_cor_semantica)

    # Configura a figura
    altura_fig = max(6, len(df_plot) * 0.4)
    fig, ax = plt.subplots(figsize=(10, altura_fig))

    # Desenha as barras baseadas na métrica selecionada
    bars = ax.barh(
        y=df_plot['Padrao'], 
        width=df_plot[col_alvo], 
        color=df_plot['cor'],
        edgecolor='none',
        height=0.6,
        alpha=0.9
    )

    # Inverte o eixo Y para o Top 1 ficar em cima
    ax.invert_yaxis()

    # Títulos Dinâmicos
    prefixo_titulo = "Padrões Sequenciais"
    if tipo_sequencia == 'precursora': prefixo_titulo += " Precursores"
    elif tipo_sequencia == 'sucessora': prefixo_titulo += " Sucessores"
    
    ax.set_title(f"{prefixo_titulo} ({conf['titulo_sufixo']})", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(conf['label_eixo'], fontsize=11, fontweight='bold')
    ax.set_ylabel("Padrão Sequencial", fontsize=11, fontweight='bold')
    
    sns.despine(left=True, bottom=True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)

    # Anotação de valores (Data Labels) com formatação específica
    for bar in bars:
        width = bar.get_width()
        label_y = bar.get_y() + bar.get_height() / 2
        
        ax.text(
            width + (width * 0.02), # Offset dinâmico baseado no valor
            label_y,
            conf['formato_label'].format(width), 
            va='center', 
            ha='left', 
            fontsize=10, 
            color='#333333',
            fontweight='bold'
        )

    # Ajuste de margens para o texto não cortar
    ax.set_xlim(0, df_plot[col_alvo].max() * 1.20)
    ax.tick_params(axis='y', labelsize=11)

    # Nome do arquivo customizado
    # Remove a extensão original se houver e adiciona o sufixo da métrica
    nome_final = f"{caminho_base.stem}_{conf['sufixo_arquivo']}.png"
    caminho_final = caminho_base.parent / nome_final

    try:
        plt.tight_layout()
        plt.savefig(caminho_final, bbox_inches='tight')
        plt.close()
        console.print(f"     [green]Gráfico ({metrica}) salvo:[/green] {nome_final}")
    except Exception as e:
        console.print(f"     [red]Erro ao salvar gráfico de {metrica}: {e}[/red]")

'''
    Função para formatar os resultados minerados e salvar em um CSV com métricas completas
    Inclui Suporte, Confiança e Lift para análise acadêmica

    @param resultados_minerados - Dicionários vindos da mineração
    @param tipo_sequencia - 'precursora' ou 'sucessora'
    @param caminho_saida - Destino do arquivo
'''
def formatar_e_salvar_resultados(resultados_minerados: list[dict], tipo_sequencia: str, caminho_saida: Path) -> None:
    if not resultados_minerados:
        console.print(f"     [yellow]Nenhum padrão relevante encontrado para {caminho_saida.name}.[/yellow]")
        return

    dados_csv = []

    for item in resultados_minerados:
        dados_csv.append({
            'Padrao': item['padrao_str'],
            'Tamanho': item['tamanho'],
            'Suporte Absoluto': item['suporte_abs'],
            'Suporte no Evento (%)': round(item['suporte_rel'], 2),
            'Confianca': item['confianca'],
            'Lift': item['lift']
        })
    
    df_resultados = pd.DataFrame(dados_csv)
    
    try:
        # 1. Persistência dos dados brutos em CSV
        df_resultados.to_csv(caminho_saida, index=False)
        console.print(f"     [green]Resultados (Suporte/Confiança/Lift) salvos:[/green] {caminho_saida.name}")
        
        # 2. Preparação para a geração de gráficos
        # Renomear para compatibilidade com a função de plotagem
        df_plot = df_resultados.rename(columns={'Suporte no Evento (%)': 'Suporte Percentual'})
        
        # O caminho base servirá de semente para os nomes dos arquivos (suporte, confianca, lift)
        caminho_base_grafico = caminho_saida.parent / f"plot_{caminho_saida.stem}"
        
        # 3. Geração dos três gráficos acadêmicos
        # Cada chamada gera um arquivo com sufixo próprio e ordenação específica
        for metrica in ['suporte', 'confianca', 'lift']:
            gerar_grafico_barras_padroes(
                df=df_plot, 
                caminho_base=caminho_base_grafico, 
                tipo_sequencia=tipo_sequencia, 
                metrica=metrica, 
                top_n=15
            )
        
    except Exception as e:
        console.print(f"[red]Erro ao processar salvamento de resultados: {e}[/red]")

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
    console.print(f"\n[bold magenta]=== Mineração de Sequências Globais ({tipo_analise.upper()}) ===[/bold magenta]")
    
    pasta_busca = BASE_DATA_FOLDER 
    if not pasta_busca.is_dir(): return

    # Itera sobre as pastas de grupos/youtubers
    for pasta_grupo in [d for d in pasta_busca.iterdir() if d.is_dir()]:
        if pasta_grupo.name == 'plots': continue
        
        nome_grupo = pasta_grupo.name
        pasta_plots = pasta_grupo / 'plots'
        pasta_plots.mkdir(exist_ok=True)

        # Processamento PRE (Causas)
        arquivo_pre = list(pasta_grupo.glob(f"sequencias_pre_{tipo_analise}_*.csv"))
        if arquivo_pre:
            caminho = arquivo_pre[0]
            console.print(f"\n[bold cyan]> Analisando CAUSAS em: {nome_grupo}[/bold cyan]")
            
            listas, rotulos = carregar_listas_para_mineracao(caminho, modo='pre')
            resultados = minerar_padroes_frequentes(listas, rotulos, min_suporte, modo='pre', min_tamanho=2)
            
            nome_csv = f"padroes_CAUSA_{tipo_analise}_{nome_grupo}.csv"
            formatar_e_salvar_resultados(resultados, 'precursora', pasta_grupo / nome_csv)
            
            gerar_sankey_sequencias(resultados, 'pre', pasta_plots / f"sankey_CAUSA_{nome_grupo}.html", f"Fluxo de Causa: {nome_grupo}")

        # Processamento POS (Consequências)
        arquivo_pos = list(pasta_grupo.glob(f"sequencias_pos_{tipo_analise}_*.csv"))
        if arquivo_pos:
            caminho = arquivo_pos[0]
            console.print(f"\n[bold cyan]> Analisando CONSEQUÊNCIAS em: {nome_grupo}[/bold cyan]")
            
            listas, rotulos = carregar_listas_para_mineracao(caminho, modo='pos')
            resultados = minerar_padroes_frequentes(listas, rotulos, min_suporte, modo='pos', min_tamanho=2)
            
            nome_csv = f"padroes_CONSEQUENCIA_{tipo_analise}_{nome_grupo}.csv"
            formatar_e_salvar_resultados(resultados, 'sucessora', pasta_grupo / nome_csv)
            
            gerar_sankey_sequencias(resultados, 'pos', pasta_plots / f"sankey_CONSEQUENCIA_{nome_grupo}.html", f"Fluxo de Consequência: {nome_grupo}")

    console.print("\n[bold green]=== Mineração de Relevância Concluída ===[/bold green]")

if __name__ == "__main__":
    MIN_SUPORTE_PERCENT = 10.0 
    
    orquestrar_mineracao_sequencias('toxicidade', MIN_SUPORTE_PERCENT)    
    #orquestrar_mineracao_sequencias('misto_9_estados', MIN_SUPORTE_PERCENT)    
    # orquestrar_mineracao_sequencias('sentimento', MIN_SUPORTE_PERCENT)