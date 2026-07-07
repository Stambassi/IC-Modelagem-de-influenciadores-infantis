import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

console = Console()

# Dicionário de configuração mantido para consistência do pipeline
METRICAS_CONFIG = {
    'detoxify': {'estados': ['NT', 'GZ', 'T']},
    'perspective': {'estados': ['NT', 'GZ', 'T']},
    'pysentimiento': {'estados': ['POS', 'NEU', 'NEG']}
}

# Configuração de Thresholds para classificação on-the-fly
CONFIG_ANALISE = {
    'detoxify': {'coluna': 'toxicity', 't_min': 0.20, 't_max': 0.80},
    'perspective': {'coluna': 'p_toxicity', 't_min': 0.20, 't_max': 0.40}
}

'''
    Função para validar e imprimir a proporção de vídeos "Non-Toxic" (sem transição para o estado tóxico alvo) em cada cluster

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param algoritmo_cluster - O algoritmo utilizado no agrupamento (ex: 'KMeans', 'DBSCAN')
    @param estado_alvo - O estado considerado como tóxico ou indesejado (padrão: 'T')
'''
def relatorio_pureza_clusters(df_vmg: pd.DataFrame, algoritmo_cluster: str, estado_alvo: str = 'T') -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    
    # Filtra as colunas que representam chegadas ao estado alvo (ex: termina com '->T')
    colunas_destino_alvo = [col for col in df_vmg.columns if col.endswith(f'->{estado_alvo}')]
    
    if not colunas_destino_alvo:
        console.print(f"[red]Erro: Nenhuma coluna de destino para o estado '{estado_alvo}' encontrada.[/red]")
        return

    # Um vídeo é considerado "puro/non-toxic" se a soma das transições para o estado T for zero
    quantidade_toxico = df_vmg[colunas_destino_alvo].sum(axis=1)
    df_vmg['is_non_toxic'] = quantidade_toxico == 0

    clusters = sorted(df_vmg[cluster_col].unique())
    total_non_toxic_geral = df_vmg['is_non_toxic'].sum()

    # Criação da Tabela para o Console
    tabela = Table(title=f"Análise de Pureza (Vídeos sem transição para '{estado_alvo}')", show_header=True, header_style="bold magenta")
    tabela.add_column("Cluster", justify="center")
    tabela.add_column("Tamanho do Cluster", justify="right")
    tabela.add_column("Qtd. Non-Toxic", justify="right")
    tabela.add_column("Pureza do Cluster (%)", justify="right", style="green")
    tabela.add_column("Representação Global (%)", justify="right", style="cyan")

    for cluster_id in clusters:
        cluster_df = df_vmg[df_vmg[cluster_col] == cluster_id]
        cluster_size = len(cluster_df)
        
        non_toxic_count = cluster_df['is_non_toxic'].sum()
        
        perc_pureza = (non_toxic_count / cluster_size * 100) if cluster_size > 0 else 0
        perc_global = (non_toxic_count / total_non_toxic_geral * 100) if total_non_toxic_geral > 0 else 0

        tabela.add_row(
            str(cluster_id),
            str(cluster_size),
            str(non_toxic_count),
            f"{perc_pureza:.2f}%",
            f"{perc_global:.2f}%"
        )

    console.print(tabela)

'''
    Função para detalhar a participação e o peso de cada youtuber dentro dos clusters gerados,
    incluindo a frequência absoluta e relativa das transições de níveis (ex: NT, GZ, T).

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param algoritmo_cluster - O algoritmo utilizado no agrupamento (ex: 'KMeans', 'DBSCAN')
    @param estados - Lista de estados da métrica configurada (padrão: ['NT', 'GZ', 'T'])
'''
def relatorio_participacao_youtubers(df_vmg: pd.DataFrame, algoritmo_cluster: str, estados: list[str] = ['NT', 'GZ', 'T']) -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    clusters = sorted(df_vmg[cluster_col].unique())
    
    # 1. Pré-calcula as visitas (chegadas) totais por estado para cada vídeo
    for estado in estados:
        colunas_destino = [col for col in df_vmg.columns if col.endswith(f'->{estado}')]
        df_vmg[f'visitas_{estado}'] = df_vmg[colunas_destino].sum(axis=1)
        
    df_vmg['visitas_totais'] = df_vmg[[f'visitas_{estado}' for estado in estados]].sum(axis=1)
    
    # Calcula o total de vídeos que cada youtuber possui em todo o escopo atual
    total_por_youtuber = df_vmg['youtuber'].value_counts()

    for cluster_id in clusters:
        cluster_df = df_vmg[df_vmg[cluster_col] == cluster_id]
        tamanho_cluster = len(cluster_df)
        
        tabela = Table(
            title=f"Participação dos Youtubers - Cluster {cluster_id} (Total: {tamanho_cluster} vídeos)", 
            show_header=True, 
            header_style="bold blue"
        )
        tabela.add_column("Youtuber", justify="left")
        tabela.add_column("Vídeos", justify="right")
        tabela.add_column("% Ocupação", justify="right", style="cyan")
        tabela.add_column("% Acervo", justify="right", style="yellow")
        
        # Cria as colunas dinâmicas para os estados com coloração semântica (se aplicável)
        cores_estados = {"NT": "green", "GZ": "grey74", "T": "red"}
        for estado in estados:
            cor = cores_estados.get(estado, "magenta")
            tabela.add_column(f"{estado} (Abs | %YT | %Clust)", justify="center", style=cor)
        
        # Conta a quantidade de vídeos por youtuber dentro deste cluster específico
        contagem_youtuber_cluster = cluster_df['youtuber'].value_counts()
        
        # Calcula o total de visitas em cada estado para o cluster inteiro (para a % do Cluster)
        total_visitas_cluster_estado = {e: cluster_df[f'visitas_{e}'].sum() for e in estados}
        
        for youtuber, count in contagem_youtuber_cluster.items():
            # Métricas em Nível de Vídeo
            perc_ocupacao_cluster = (count / tamanho_cluster) * 100
            perc_acervo_youtuber = (count / total_por_youtuber[youtuber]) * 100
            
            row_data = [
                str(youtuber),
                str(count),
                f"{perc_ocupacao_cluster:.2f}%",
                f"{perc_acervo_youtuber:.2f}%"
            ]
            
            # Filtra os dados apenas desse youtuber dentro desse cluster
            yt_cluster_df = cluster_df[cluster_df['youtuber'] == youtuber]
            visitas_yt_total = yt_cluster_df['visitas_totais'].sum()
            
            # Métricas em Nível de Transição (NT, GZ, T)
            for estado in estados:
                visitas_yt_estado = yt_cluster_df[f'visitas_{estado}'].sum()
                
                # Frequência Relativa ao Youtuber (quanto % das falas DELE são desse estado)
                perc_yt = (visitas_yt_estado / visitas_yt_total * 100) if visitas_yt_total > 0 else 0
                
                # Frequência Relativa ao Cluster (quanto % das falas do CLUSTER nesse estado pertencem a ele)
                total_estado_cluster = total_visitas_cluster_estado[estado]
                perc_clust = (visitas_yt_estado / total_estado_cluster * 100) if total_estado_cluster > 0 else 0
                
                # Ajusta a exibição absoluta caso seja contagem pura ou probabilidade fracionada
                val_abs = int(visitas_yt_estado) if visitas_yt_estado % 1 == 0 else f"{visitas_yt_estado:.2f}"
                
                formato_estado = f"{val_abs} ({perc_yt:.1f}% | {perc_clust:.1f}%)"
                row_data.append(formato_estado)
                
            tabela.add_row(*row_data)

        console.print(tabela)

'''
    Função para ranquear os youtubers criando um "Índice de Volatilidade" baseado na sua proporção de vídeos no cluster mais tóxico

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param algoritmo_cluster - O algoritmo utilizado no agrupamento (ex: 'KMeans', 'DBSCAN')
    @param estado_alvo - O estado considerado como tóxico ou indesejado (padrão: 'T')
'''
def relatorio_volatilidade_youtubers(df_vmg: pd.DataFrame, algoritmo_cluster: str, estado_alvo: str = 'T') -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    
    # 1. Descobre qual é o "Cluster Tóxico" (baseado no volume de toxicidade)
    colunas_destino_alvo = [col for col in df_vmg.columns if col.endswith(f'->{estado_alvo}')]
    if not colunas_destino_alvo: 
        return
    
    # Soma todas as chegadas ao estado tóxico
    df_vmg['intensidade_toxica'] = df_vmg[colunas_destino_alvo].sum(axis=1)
    
    # Calcula a média de transições tóxicas por vídeo dentro de cada cluster
    media_toxicidade_cluster = df_vmg.groupby(cluster_col)['intensidade_toxica'].mean()
    
    # Elege o cluster mais volátil como aquele que possui a maior média de intensidade tóxica
    cluster_toxico = media_toxicidade_cluster.idxmax() 
    
    # 2. Calcula o Índice de Volatilidade para cada Youtuber
    total_por_youtuber = df_vmg['youtuber'].value_counts()
    df_toxico = df_vmg[df_vmg[cluster_col] == cluster_toxico]
    toxico_por_youtuber = df_toxico['youtuber'].value_counts()
    
    dados_ranking = []
    for youtuber, total in total_por_youtuber.items():
        qtd_toxico = toxico_por_youtuber.get(youtuber, 0)
        indice_volatilidade = (qtd_toxico / total) * 100
        dados_ranking.append((youtuber, total, qtd_toxico, indice_volatilidade))
        
    # Ordena do mais volátil para o mais puro
    dados_ranking.sort(key=lambda x: x[3], reverse=True)
    
    # 3. Exibição da Tabela
    tabela = Table(title=f"Índice de Volatilidade (Ocupação no Cluster Tóxico: {cluster_toxico})", show_header=True, header_style="bold red")
    tabela.add_column("Posição", justify="center")
    tabela.add_column("Youtuber", justify="left")
    tabela.add_column("Acervo Total", justify="right")
    tabela.add_column(f"Vídeos no Cluster {cluster_toxico}", justify="right")
    tabela.add_column("Índice de Volatilidade (%)", justify="right", style="red")
    
    for idx, (youtuber, total, qtd_toxico, iv) in enumerate(dados_ranking, 1):
        tabela.add_row(str(idx), youtuber, str(total), str(qtd_toxico), f"{iv:.2f}%")
        
    console.print(tabela)

'''
    Função para extrair e apresentar o contexto qualitativo das Segmented Video Transcriptions (SVTs) onde ocorrem transições chave.

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param youtubers_alvo - Lista de youtubers a serem analisados qualitativamente (ex: ['Athos', 'JP Plays'])
    @param nome_analise - O nome da análise a ser lida (ex: 'detoxify', 'perspective')
    @param algoritmo_cluster - O algoritmo de clusterização utilizado (ex: 'KMeans', 'DBSCAN')
    @param transicoes_alvo - Lista de transições a buscar (ex: ['NT->T', 'GZ->T'])
    @param max_videos - Número máximo de vídeos amostrados por youtuber
'''
def relatorio_contexto_qualitativo(
    df_vmg: pd.DataFrame, 
    youtubers_alvo: list[str], 
    nome_analise: str, 
    algoritmo_cluster: str,
    transicoes_alvo: list[str] = ['NT->T', 'GZ->T', 'T->T'],
    max_videos: int = 3
) -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    
    console.print(f"[bold yellow]🔍 Iniciando Extração Qualitativa de SVTs para: {', '.join(youtubers_alvo)}[/bold yellow]")
    
    for youtuber in youtubers_alvo:
        df_yt = df_vmg[df_vmg['youtuber'] == youtuber].copy()
        if df_yt.empty: 
            continue
            
        # Ordena os vídeos para pegar os que têm mais ocorrências das transições alvo
        colunas_soma = [col for col in transicoes_alvo if col in df_yt.columns]
        if colunas_soma:
            df_yt['score_alvo'] = df_yt[colunas_soma].sum(axis=1)
            df_yt = df_yt.sort_values(by='score_alvo', ascending=False)
            
        # Pega os Top vídeos com maior carga das transições buscadas
        videos_selecionados = df_yt[df_yt['score_alvo'] > 0].head(max_videos)
        
        if videos_selecionados.empty:
            console.print(f"[dim]Nenhuma transição alvo encontrada nos vídeos de {youtuber}.[/dim]")
            continue

        for _, row in videos_selecionados.iterrows():
            video_id = row['video_id']
            cluster_id = row[cluster_col]
            
            base_yt_path = Path(f'files/{youtuber}')

            tiras_files = [p for p in base_yt_path.rglob('tiras_video.csv') if p.parent.name == video_id]
            transicoes_files = [p for p in base_yt_path.rglob(f'transicoes_{nome_analise}.csv') if p.parent.parent.parent.name == video_id]

            if not tiras_files or not transicoes_files:
                console.print(f"[yellow]Aviso: Arquivos brutos não encontrados para o vídeo {video_id}.[/yellow]")
                continue
                
            caminho_tiras = tiras_files[0]
            caminho_transicoes = transicoes_files[0]
            
            df_transicoes = pd.read_csv(caminho_transicoes)
            df_tiras = pd.read_csv(caminho_tiras)
            
            coluna_texto = 'tiras' if 'tiras' in df_tiras.columns else None
            if not coluna_texto:
                console.print(f"[yellow]Aviso: Coluna 'tiras' não encontrada em {caminho_tiras.name}.[/yellow]")
                continue

            # Identifica as transições desejadas
            df_transicoes['transicao_atual'] = df_transicoes['estado'].astype(str) + '->' + df_transicoes['proximo_estado'].astype(str)
            indices_alvo = df_transicoes[df_transicoes['transicao_atual'].isin(transicoes_alvo)].index.tolist()

            if not indices_alvo:
                continue
                
            console.print(f"\n[bold cyan]👤 Youtuber:[/bold cyan] {youtuber} | [bold cyan]🎬 Vídeo:[/bold cyan] {video_id} | [bold cyan]🧩 Cluster:[/bold cyan] {cluster_id}")
            
            # Pega as 3 primeiras ocorrências no vídeo para o relatório
            for idx in indices_alvo[:3]:
                transicao_ocorrida = df_transicoes.loc[idx, 'transicao_atual']
                console.print(f"[bold red] ⚡ Gatilho Identificado:[/bold red] {transicao_ocorrida} (Linha {idx})")
                
                # Se a transição ocorreu no índice 'idx' (tempo t para t+1),
                # a SVT de origem está na linha 'idx' e a SVT de destino na 'idx + 1' do arquivo de tiras.
                for offset, label in [(0, "SVT Anterior"), (1, "SVT da Transição")]:
                    linha_tira = idx + offset
                    if linha_tira < len(df_tiras):
                        estado_str = df_transicoes.loc[idx, 'estado'] if offset == 0 else df_transicoes.loc[idx, 'proximo_estado']
                        texto_svt = df_tiras.loc[linha_tira, coluna_texto]
                        console.print(f"    [dim]{label} (Estado {estado_str}):[/dim] '{texto_svt}'")
                
                console.print("    [dim]" + "-"*60 + "[/dim]")

'''
    Função para gerar nuvens de palavras (WordClouds) das Segmented Video Transcriptions (SVTs).
    Utiliza spaCy para lematização e filtro morfológico (Verbos, Adjetivos, Substantivos) e 
    TF-IDF para extrair termos exclusivos de cada estado emocional.

    @param df_vmg - DataFrame contendo metadados e rotulagem dos vídeos/clusters
    @param youtubers_alvo - Lista de youtubers para gerar as wordclouds
    @param nome_analise - O nome da análise (ex: 'detoxify', 'perspective')
    @param estados - Lista de estados da métrica configurada (ex: ['NT', 'GZ', 'T'])
    @param base_dir_agrupamento - Caminho base da pasta de agrupamento atual para salvar as imagens
'''
def relatorio_wordcloud(
    df_vmg: pd.DataFrame, 
    youtubers_alvo: list[str], 
    nome_analise: str, 
    estados: list[str], 
    base_dir_agrupamento: Path
) -> None:
    
    # 1. Carregamento do Modelo Linguístico (spaCy)
    try:
        nlp = spacy.load('pt_core_news_md', disable=['parser', 'ner'])
    except OSError:
        console.print("[red]Erro: Modelo do spaCy não encontrado. Execute: python -m spacy download pt_core_news_sm[/red]")
        return

    console.print(f"[bold yellow]☁️ Iniciando Geração de WordClouds Avançadas (spaCy + TF-IDF) para {len(youtubers_alvo)} youtubers...[/bold yellow]")
    
    # Custom stopwords pesadas para a oralidade gamer
    custom_stopwords = {
        'né', 'tá', 'pra', 'pro', 'aí', 'aqui', 'lá', 'então', 'assim', 'vai', 'vou', 'um', 'uma', 
        'ele', 'ela', 'isso', 'aquilo', 'ser', 'gente', 'ah', 'tô', 'to', 'ai', 'acho', 'ó', 'ok', 
        'mim', 'deu', 'agora', 'olha', 'pessoal', 'cara', 'mano', 'velho', 'galera', 'fazer', 'ter', 'ir'
    }
    
    CONFIG_ANALISE = {
        'detoxify': {'coluna': 'toxicity', 't_min': 0.20, 't_max': 0.80},
        'perspective': {'coluna': 'p_toxicity', 't_min': 0.20, 't_max': 0.40}
    }
    
    config_atual = CONFIG_ANALISE.get(nome_analise)
    if not config_atual:
        console.print(f"[red]Erro: Configuração de thresholds não encontrada para '{nome_analise}'.[/red]")
        return

    # Helper function para filtrar e lematizar os textos com spaCy
    def processar_textos_spacy(lista_textos):
        lemmas = []
        # O nlp.pipe processa os textos em blocos (batch_size), ignorando o limite de tamanho total
        for doc in nlp.pipe(lista_textos, batch_size=100):
            for token in doc:
                # Filtro Triplo: Tamanho da palavra, Stopwords e Classe Gramatical
                if len(token.text) > 2 and not token.is_stop and token.text.lower() not in custom_stopwords:
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        lemmas.append(token.lemma_.lower())
        return " ".join(lemmas)

    for youtuber in youtubers_alvo:
        df_yt = df_vmg[df_vmg['youtuber'] == youtuber]
        if df_yt.empty: continue
        
        # Coleta de strings cruas
        textos_brutos = {est: [] for est in estados}
        textos_brutos['Total'] = []
            
        videos_processados = 0
            
        for video_id in df_yt['video_id'].unique():
            base_yt_path = Path(f'files/{youtuber}')
            tiras_files = [p for p in base_yt_path.rglob('tiras_video.csv') if p.parent.name == video_id]
            if not tiras_files: continue
                
            df_tiras = pd.read_csv(tiras_files[0])
            coluna_texto = 'tiras' if 'tiras' in df_tiras.columns else None
            coluna_score = config_atual['coluna']
            
            if not coluna_texto or coluna_score not in df_tiras.columns: continue
            
            for _, row in df_tiras.dropna(subset=[coluna_texto, coluna_score]).iterrows():
                texto = str(row[coluna_texto]).strip()
                if not texto or texto == 'nan': continue
                    
                score = float(row[coluna_score])
                
                # Classificação On-The-Fly
                if score < config_atual['t_min']: estado_atual = 'NT'
                elif score >= config_atual['t_max']: estado_atual = 'T'
                else: estado_atual = 'GZ'
                
                textos_brutos['Total'].append(texto)
                textos_brutos[estado_atual].append(texto)
                    
            videos_processados += 1
            
        if videos_processados == 0: continue

        # 2. Processamento Linguístico (Lematização em Lotes)
        textos_processados = {}
        for categoria, lista_textos in textos_brutos.items():
            # Verifica se há um volume mínimo de textos para valer a pena processar
            if len(lista_textos) > 5:
                # Passa a lista diretamente para a nova função
                textos_processados[categoria] = processar_textos_spacy(lista_textos)
            else:
                textos_processados[categoria] = ""

        # 3. Modelagem TF-IDF
        # Cria um corpus apenas com os estados (NT, GZ, T) para gerar contraste entre eles.
        # Não coloca o 'Total' no TF-IDF pois ele é a soma das partes e prejudicaria a matemática do contraste.
        corpus_estados = [textos_processados[est] for est in estados if textos_processados[est]]
        
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus_estados)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Caso não haja vocabulário suficiente após a limpeza
            console.print(f"[dim]Vocabulário insuficiente após limpeza para {youtuber}.[/dim]")
            continue
            
        output_dir = base_dir_agrupamento / 'Word Cloud'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        wc_config = {
            'width': 800, 'height': 400, 
            'background_color': 'white',
            'max_words': 150
        }
        
        # 4. Geração das Imagens
        # Gera as nuvens de contraste (NT, GZ, T) usando as frequências do TF-IDF
        idx_matriz = 0
        for estado in estados:
            texto_lema = textos_processados[estado]
            if not texto_lema: continue
                
            # Extrai os pesos TF-IDF específicos para este documento (estado)
            tfidf_scores = tfidf_matrix[idx_matriz].toarray()[0]
            word_freqs = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
            idx_matriz += 1
            
            if not word_freqs: continue
            
            if estado == 'T': wc_config['colormap'] = 'Reds'
            elif estado == 'GZ': wc_config['colormap'] = 'copper'
            elif estado == 'NT': wc_config['colormap'] = 'Greens'
            
            try:
                wc = WordCloud(**wc_config).generate_from_frequencies(word_freqs)
                wc.to_file(str(output_dir / f'wc_{nome_analise}_{estado}.png'))
            except Exception as e:
                console.print(f"[red]Erro WordCloud '{estado}' - {youtuber}: {e}[/red]")

        # Gera a nuvem Total usando apenas a lematização bruta (sem o contraste do TF-IDF)
        # para mostrar o "Vibe" geral do canal
        texto_total = textos_processados.get('Total', '')
        if len(texto_total) > 10:
            wc_config['colormap'] = 'Blues'
            try:
                wc = WordCloud(**wc_config).generate(texto_total)
                wc.to_file(str(output_dir / f'wc_{nome_analise}_Total.png'))
            except Exception as e:
                pass
                
        console.print(f"    [green]✓ WordClouds (spaCy+TF-IDF) salvas para [bold]{youtuber}[/bold] ({videos_processados} vídeos)[/green]")

'''
    Função principal para orquestrar a leitura dos dados e a execução das análises de perfil dos clusters parametrizada.

    @param escopo - O alvo do agrupamento a ser analisado (ex: 'Geral', 'Minecraft', 'Roblox')
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param nome_analise - O nome da análise a ser lida (ex: 'detoxify', 'perspective')
    @param metrica_agrupamento - A métrica que originou o agrupamento atual (ex: 'contagem', 'probabilidade')
    @param algoritmo - O algoritmo de clusterização utilizado (ex: 'KMeans', 'DBSCAN')
    @param tipo_analise - O relatório a ser gerado ('pureza', 'youtubers', 'volatilidade', 'qualitativa', 'todas')
'''
def analisar_caracteristicas_clusters(
    escopo: str, 
    mapa_categorias: dict, 
    nome_analise: str, 
    metrica_agrupamento: str, 
    algoritmo: str,
    tipo_analise: str = 'todas'
) -> None:
    console.print(f"[bold blue]>>> Analisando '{tipo_analise.capitalize()}' - Escopo: {escopo} ({nome_analise} | {metrica_agrupamento})[/bold blue]\n")

    # 1. Resolução Dinâmica de Diretórios
    if escopo == 'Geral' or escopo in mapa_categorias.values():
        base_dir_agrupamento = Path(f'files/VMG/{escopo}/Agrupamento')
    else:
        base_dir_agrupamento = Path(f'files/{escopo}/VMG/Agrupamento')

    csv_cluster_path = base_dir_agrupamento / f'cluster_{nome_analise}_{metrica_agrupamento}_{algoritmo.lower()}.csv'
    
    if not csv_cluster_path.exists():
        console.print(f"[yellow]Aviso: Arquivo {csv_cluster_path.name} não encontrado. Execute o agrupamento primeiro.[/yellow]")
        return

    # 2. Leitura dos Dados
    df_vmg = pd.read_csv(csv_cluster_path)
    
    estados_config = METRICAS_CONFIG.get(nome_analise, {}).get('estados', ['NT', 'GZ', 'T'])
    estado_toxico_alvo = estados_config[-1]
    # estado_toxico_alvo = estados_config[-2]

    # 3. Execução das Análises Parametrizadas
    if tipo_analise in ['pureza', 'todas']:
        relatorio_pureza_clusters(df_vmg, algoritmo, estado_alvo=estado_toxico_alvo)
        
    if tipo_analise in ['youtubers', 'todas']:
        relatorio_participacao_youtubers(df_vmg, algoritmo, estados=estados_config)
        
    if tipo_analise in ['volatilidade', 'todas']:
        relatorio_volatilidade_youtubers(df_vmg, algoritmo, estado_alvo=estado_toxico_alvo)
        
    if tipo_analise in ['qualitativa', 'todas']:
        relatorio_contexto_qualitativo(
            df_vmg=df_vmg, 
            youtubers_alvo=['Athos', 'JP Plays'], 
            nome_analise=nome_analise, 
            algoritmo_cluster=algoritmo,
            transicoes_alvo=[f'NT->{estado_toxico_alvo}', f'GZ->{estado_toxico_alvo}', f'{estado_toxico_alvo}->{estado_toxico_alvo}']
        )

    if tipo_analise in ['wordcloud', 'todas']:
        lista_todos_youtubers = df_vmg['youtuber'].unique().tolist()
        relatorio_wordcloud(
            df_vmg=df_vmg,
            youtubers_alvo=lista_todos_youtubers,
            nome_analise=nome_analise,
            estados=estados_config,
            base_dir_agrupamento=base_dir_agrupamento
        )

if __name__ == "__main__":
    # Mapeia cada youtuber para sua categoria principal
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

    # Setup das escolhas
    # escopos = ['Julia MineGirl']
    # escopos = ['Geral', 'Roblox', 'Minecraft']
    escopos = list(mapa_youtubers_categoria.keys())

    analises = ['perspective', 'detoxify']
    algoritmos = ['KMeans']
    
    metrica_base = 'contagem' 

    for escopo in escopos:
        for analise in analises:
            for alg in algoritmos:
                analisar_caracteristicas_clusters(
                    escopo=escopo,
                    mapa_categorias=mapa_youtubers_categoria,
                    nome_analise=analise,
                    metrica_agrupamento=metrica_base,
                    algoritmo=alg,
                    # tipo_analise='pureza'
                    # tipo_analise='youtubers'
                    # tipo_analise='volatilidade'
                    # tipo_analise='qualitativa'
                    tipo_analise='wordcloud'
                )