from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

console = Console()

'''
    Função para armazenar as transições de estados de sentimento (POS, NEU, NEG) 
    de cada vídeo em um arquivo CSV.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_transicoes_sentimento(youtubers_list: list[str]) -> None:
    # Nomes dos estados de sentimento
    estados_sentimento = ['POS', 'NEU', 'NEG']
    # Colunas que espera-se encontrar no CSV de entrada
    colunas_sentimento = ['toxicidade', 'positividade', 'neutralidade']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando transições de sentimento para [bold cyan]{youtuber}[/bold cyan]')

        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            video_path = tiras_csv_path.parent
            
            try:
                (video_path / 'transicoes').mkdir(parents=True, exist_ok=True)
                
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se o DataFrame tem as colunas necessárias e dados suficientes
                if df_tiras_video.empty or not all(col in df_tiras_video.columns for col in colunas_sentimento) or len(df_tiras_video) < 2:
                    continue

                # Usar a coluna 'grupo' como a coluna de 'estado'
                df_tiras_video.rename(columns={'grupo': 'estado'}, inplace=True)
                
                # Para robustez, garantimos que apenas os valores esperados ('POS', 'NEU', 'NEG') sejam considerados
                df_tiras_video = df_tiras_video[df_tiras_video['estado'].isin(estados_sentimento)]
                
                # Re-verificar se ainda há dados suficientes após a filtragem
                if len(df_tiras_video) < 2:
                    continue
                
                # Converter a coluna 'estado' para o tipo Categórico para que o groupby gere todas as 3x3=9 transições
                tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
                df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)

                # Lógica de transição
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)
                
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa e conta, incluindo as transições que não ocorreram (contagem 0)
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')

                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Salva no novo arquivo CSV com nome descritivo
                output_path = video_path / 'transicoes' / 'transicoes_sentimento.csv'
                contagem.to_csv(output_path, index=False)
            
            except Exception as e:
                console.print(f'[bold red]Erro[/bold red] em {video_path.name} (salvar_transicoes_sentimento): {e}')

'''
    Função para criar e persistir a Matriz de Transição de Sentimento (POS, NEU, NEG) para cada vídeo.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_matriz_transicao_sentimento_video(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes de sentimento para [bold cyan]{youtuber}[/bold cyan]')

        # Procurar recursivamente pelo arquivo de transições de sentimento
        for transicoes_csv_path in base_path.rglob('transicoes_sentimento.csv'):                
            try:                
                df_transicoes = pd.read_csv(transicoes_csv_path)

                if df_transicoes.empty:
                    continue

                # Para garantir que a matriz final seja sempre 3x3, mesmo que um estado nunca apareça como ponto de partida, converte-se as colunas para o tipo Categórico.
                estados_sentimento = ['POS', 'NEU', 'NEG']
                tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
                df_transicoes['estado'] = df_transicoes['estado'].astype(tipo_categorico)
                df_transicoes['proximo_estado'] = df_transicoes['proximo_estado'].astype(tipo_categorico)

                # Calcular a soma das transições que saem de cada estado
                # # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence. 
                somas_por_estado = df_transicoes.groupby('estado', observed=False)['contagem'].transform('sum')

                # Calcular a probabilidade de cada transição 
                probabilidade = df_transicoes['contagem'] / somas_por_estado

                # Se a soma for 0 (um estado nunca foi visitado), o resultado da divisão será NaN (Not a Number)
                # Usa-se fillna(0) para tratar esses casos, definindo a probabilidade como 0.
                df_transicoes['probabilidade'] = (probabilidade).fillna(0) 

                # Transforma o formato "longo" (uma transição por linha) para o formato "largo" (matriz),
                # onde o índice são os estados de origem, as colunas são os estados de destino,
                # e os valores são as probabilidades calculadas. 
                matriz_transicao = df_transicoes.pivot(
                    index='estado', 
                    columns='proximo_estado', 
                    values='probabilidade'
                )
                
                # Se algum estado nunca foi ponto de partida, o pivot pode criar NaNs, então garante-se que sejam 0
                matriz_transicao.fillna(0, inplace=True)
                
                # Salvar a matriz
                output_path = transicoes_csv_path.parent / 'matriz_transicao_sentimento.csv'
                matriz_transicao.to_csv(output_path)

            except Exception as e:
                video_path = transicoes_csv_path.parent.parent
                console.print(f'Inválido (salvar_matriz_transicao_sentimento_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição de Sentimento agregada para cada youtuber.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_matriz_transicao_sentimento_youtuber(youtubers_list: list[str]) -> None:
    # Nomes dos estados de sentimento
    estados_sentimento = ['POS', 'NEU', 'NEG']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matriz de sentimento agregada para [bold cyan]{youtuber}[/bold cyan]')

        try:
            # Encontrar e concatenar todas as transições do youtuber
            lista_dfs_transicoes = []
            for transicoes_csv_path in base_path.rglob('transicoes_sentimento.csv'):
                df_video = pd.read_csv(transicoes_csv_path)
                if not df_video.empty:
                    lista_dfs_transicoes.append(df_video)
            
            if not lista_dfs_transicoes:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições de sentimento encontrado para {youtuber}. Pulando.[/yellow]")
                continue
            
            df_agregado = pd.concat(lista_dfs_transicoes, ignore_index=True)

            # Agrupar por cada tipo de transição e somar as contagens de todos os vídeos
            df_soma_total = df_agregado.groupby(['estado', 'proximo_estado'])['contagem'].sum().reset_index()

            # Garantir que a matriz final seja sempre 3x3, mesmo que algumas transições ou estados nunca tenham ocorrido no agregado de vídeos do youtuber
            tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
            df_soma_total['estado'] = df_soma_total['estado'].astype(tipo_categorico)
            df_soma_total['proximo_estado'] = df_soma_total['proximo_estado'].astype(tipo_categorico)
            
            # Calcular a soma das transições que saem de cada estado
            # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence.
            somas_por_estado = df_soma_total.groupby('estado', observed=False)['contagem'].transform('sum')

            # Calcular a probabilidade de cada transição
            probabilidade = df_soma_total['contagem'] / somas_por_estado

            # Se a soma for 0 (um estado nunca foi visitado), o resultado da divisão será NaN (Not a Number)
            # Usa-se fillna(0) para tratar esses casos, definindo a probabilidade como 0.
            df_soma_total['probabilidade'] = probabilidade.fillna(0)
            
            matriz_transicao_youtuber = df_soma_total.pivot(
                index='estado', 
                columns='proximo_estado', 
                values='probabilidade'
            )
            
            # Garantir que qualquer célula que possa ter ficado como NaN (no caso de um estado nunca ser ponto de partida) seja preenchida com 0.
            matriz_transicao_youtuber.fillna(0, inplace=True)
            
            # Salvar na pasta raiz do youtuber para consistência com outras análises agregadas
            output_path = base_path / 'transicoes' / 'matriz_transicao_sentimento_youtuber.csv'
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição de youtuber gerada: {output_path}")

        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_sentimento_youtuber): {e}')

'''
    Função para calculr a distribuição estacionária de uma Cadeia de Markov a partir de sua matriz de transição
    @param matriz_transicao - Matriz de transição n x n, em que as linhas somam 1
    @return distribuicao_estacionaria - Vetor de probabilidades com a distribuição estacionária 'pi' de um youtuber 
'''
def calcular_distribuicao_estacionaria(matriz_transicao: pd.DataFrame) -> np.ndarray:
    # Converter o DataFrame do pandas para uma matriz NumPy
    matriz_np = matriz_transicao.to_numpy()

    # A equação é pi * matriz_np = pi. Isso significa que pi é um autovetor à esquerda de matriz_np com autovalor 1.
    # Achar autovetores à esquerda de matriz_np é o mesmo que achar autovetores à direita da transposta de matriz_np (matriz_np.T).
    autovalores, autovetores = np.linalg.eig(matriz_np.T)

    # Encontrar o índice do autovetor correspondente ao autovalor mais próximo de 1
    # Usa-se np.isclose para lidar com pequenas imprecisões de ponto flutuante
    indice_estacionario = np.where(np.isclose(autovalores, 1))[0][0]
    
    # Selecionar o autovetor correspondente
    vetor_estacionario = autovetores[:, indice_estacionario]

    # O autovetor pode ter componentes complexas (com parte imaginária ~0), então pegamos apenas a parte real
    vetor_estacionario_real = np.real(vetor_estacionario)

    # Normalizar o vetor para que a soma de suas componentes seja 1, transformando-o em um vetor de probabilidade
    distribuicao_estacionaria = vetor_estacionario_real / np.sum(vetor_estacionario_real)

    return distribuicao_estacionaria

'''
    Função para carregar a matriz de transição de sentimento de cada youtuber, calcular sua distribuição estacionária e persistir os resultados
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def analisar_distribuicao_estacionaria_sentimento(youtubers_list: list[str]) -> None:
    console.print("\n--- Análise da Distribuição Estacionária de Sentimento ---", style="bold magenta")
    
    # Dicionário para armazenar os resultados
    resultados = {}

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / 'matriz_transicao_sentimento_youtuber.csv'

        if not matriz_path.exists():
            continue

        try:
            # Carregar a matriz, definindo a primeira coluna como índice
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            
            # Calcular a distribuição usando a função auxiliar
            distribuicao = calcular_distribuicao_estacionaria(matriz_df)
            
            # Armazenar o resultado junto com os nomes dos estados (ex: ['POS', 'NEU', 'NEG']) para usar como cabeçalho do DataFrame final
            resultados[youtuber] = {'distribuicao': distribuicao, 'estados': matriz_df.columns.tolist()}

        except Exception as e:
            console.print(f"Inválido (analisar_distribuicao_estacionaria_sentimento): {e}")
    
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Preparar dados para o DataFrame final
    dados_para_df = {yt: res['distribuicao'] for yt, res in resultados.items()}

    # Extrair dinamicamente os nomes das colunas dos dados
    nomes_colunas = next(iter(resultados.values()))['estados']

    # Criar um DataFrame com os resultados para fácil visualização e comparação
    df_resultados = pd.DataFrame.from_dict(dados_para_df, orient='index', columns=nomes_colunas)
    
    # Ordenar pelo estado 'NEG', que é o análogo ao estado mais tóxico
    if 'NEG' in df_resultados.columns:
        df_resultados = df_resultados.sort_values(by='NEG', ascending=False)

    # Definir o caminho para o arquivo de saída
    output_dir = Path('files/transicoes')

    # Garantir que o diretório 'files/transicoes' exista
    output_dir.mkdir(exist_ok=True) 
    
    output_path = output_dir / 'distribuicao_estacionaria_sentimento_geral.csv'

    df_resultados.to_csv(output_path, index=True, index_label='youtuber')

    console.print(f"\nResultados da distribuição estacionária de sentimento salvos em: [green]{output_path}[/green]")


'''
    Função para calcular o Tempo Médio de Permanência (Sojourn Time) para cada estado
    @param matriz_transicao - Matriz de transição n x n, em que as linhas somam 1
    @return resultado_series - Série com o tempo médio de permanência para cada estado
'''
def calcular_tempo_permanencia(matriz_transicao: pd.DataFrame) -> pd.Series:
    # Extrair os valores da diagonal principal (P_11, P_22, ...)
    p_ii = np.diag(matriz_transicao)
    
    # Calcula '1 - P_ii'. Adiciona um valor muito pequeno (epsilon) para evitar divisão por zero caso P_ii seja exatamente 1, o que resultaria em tempo infinito.
    # Usamos np.errstate para suprimir avisos sobre divisão por zero.
    with np.errstate(divide='ignore'):
        tempos_permanencia = 1 / (1 - p_ii)
    
    # Onde o tempo de permanência for infinito (P_ii=1), numpy retornará 'inf'.
    # Isso é matematicamente correto para um estado absorvente.
    
    # Cria uma série do pandas para um resultado mais claro, usando o índice da matriz (nomes dos estados)
    resultado_series = pd.Series(tempos_permanencia, index=matriz_transicao.index, name='Tempo de Permanencia (min)')
    
    return resultado_series

'''
    Função para ler a matriz de transição de sentimento de cada youtuber, calcular o tempo médio de permanência e persistir os resultados
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def analisar_tempo_permanencia_sentimento(youtubers_list: list[str]) -> None:
    console.print("\n--- Análise do Tempo Médio de Permanência de Sentimento ---", style="bold magenta")
    
    resultados = {}

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / 'matriz_transicao_sentimento_youtuber.csv'

        if not matriz_path.exists():
            continue

        try:
            matriz_df = pd.read_csv(matriz_path, index_col=0)

            tempos = calcular_tempo_permanencia(matriz_df)

            resultados[youtuber] = tempos
        except Exception as e:
            console.print(f"Inválido (analisar_tempo_permanencia_sentimento): {e}")
    
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Converter o dicionário de Séries em DataFrame. As colunas serão nomeadas automaticamente com base no índice das Séries ('POS', 'NEU', 'NEG').
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    
    # Ordenar pelo tempo de permanência no estado 'NEG'
    if 'NEG' in df_resultados.columns:
        df_resultados = df_resultados.sort_values(by='NEG', ascending=False)
    
    # Persistir os dados
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'tempo_permanencia_sentimento_geral.csv'
    df_resultados.to_csv(output_path, index=True, index_label='youtuber')

    console.print(f"\nResultados do tempo de permanência de sentimento salvos em: [green]{output_path}[/green]")

'''
    Função para carregar as matrizes de transição agregadas de cada youtuber, achatá-las em um vetor de características e persistir o resultado
    @param youtubers_list - Lista de youtubers a serem analisados
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
    @return df_vetores - Um DataFrame onde cada linha é um youtuber e as colunas são as probabilidades da matriz de transição achatada
'''
def criar_vetores_de_caracteristicas_achatados_youtuber(youtubers_list: list[str], tipo_analise: str, n: int = None) -> pd.DataFrame:
    # Testar o tipo de análise
    if tipo_analise not in ['sentimento', 'toxicidade']:
        raise ValueError("O parâmetro 'tipo_analise' deve ser 'sentimento' ou 'toxicidade'.")
    if tipo_analise == 'toxicidade' and n is None:
        raise ValueError("O parâmetro 'n' é obrigatório para a análise de 'toxicidade'.")

    console.print(f"\n--- Criando Vetores Achatados para Análise de '{tipo_analise.title()}' ---", style="bold magenta")
    
    vetores_youtubers = {}

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        
        # Definir o nome do arquivo da matriz com base no tipo de análise
        if tipo_analise == 'sentimento':
            matriz_filename = 'matriz_transicao_sentimento_youtuber.csv'
        else: # toxicidade
            matriz_filename = f'matriz_transicao_youtuber_{n}.csv'
            
        matriz_path = base_path / 'transicoes' / matriz_filename

        if not matriz_path.exists():
            console.print(f"[yellow]Aviso: Matriz não encontrada para {youtuber} em {matriz_path}. Pulando.[/yellow]")
            continue

        try:
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            
            # Converter a matriz para numpy e "achatar" em um vetor unidimensional
            vetor_achatado = matriz_df.to_numpy().flatten()
            
            vetores_youtubers[youtuber] = vetor_achatado

        except Exception as e:
            console.print(f"Inválido (criar_vetores_de_caracteristicas_achatados_youtuber): {e}")

    if not vetores_youtubers:
        console.print("[bold red]Nenhum vetor de características foi gerado. Abortando.[/bold red]")
        return None

    # Converter o dicionário de vetores em um DataFrame
    df_vetores = pd.DataFrame.from_dict(vetores_youtubers, orient='index')

    # Criar nomes de colunas descritivos (ex: P_POS_NEU, P_1_2)
    estados = matriz_df.index.tolist()

    nomes_colunas = [f'P_{origem}_para_{destino}' for origem in estados for destino in estados]

    df_vetores.columns = nomes_colunas
    
    # Salvar o DataFrame consolidado
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True)

    if tipo_analise == 'sentimento':
        output_filename = 'vetores_achatados_sentimento.csv'
    else:
        output_filename = f'vetores_achatados_toxicidade_{n}.csv'
    
    output_path = output_dir / output_filename
    df_vetores.to_csv(output_path, index=True, index_label='youtuber')
    console.print(f"\nVetores de características salvos em: [green]{output_path}[/green]")
    
    return df_vetores

'''
    Função para aplicar o escalonamento e o PCA aos vetores de características e gerar um gráfico de dispersão 2D
    @param df_vetores - DataFrame de vetores de características
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
'''
def visualizar_pca_youtuber(df_vetores: pd.DataFrame, tipo_analise: str, n: int = None) -> None:
    # Testar se o vetor de características é válido 
    if df_vetores is None or df_vetores.shape[0] < 2:
        console.print("[bold yellow]Aviso: Dados insuficientes para a análise de PCA (necessário ao menos 2 youtubers).[/bold yellow]")
        return
        
    console.print("\n--- Gerando Visualização PCA ---", style="bold magenta")

    # Escalonar dos dados para garantir que todas as características (probabilidades) tenham a mesma importância
    scaler = StandardScaler()
    dados_escalados = scaler.fit_transform(df_vetores)
    
    # Aplicar o PCA, reduzindo a dimensionalidade para apenas 2 componentes principais para visualização
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(dados_escalados)

    # Criar um DataFrame com os resultados do PCA para facilitar a plotagem
    df_pca = pd.DataFrame(data=componentes_principais, columns=['PC1', 'PC2'], index=df_vetores.index)
    
    # Gerar o gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        data=df_pca, 
        s=100,  # Tamanho dos pontos
        alpha=0.8,
        ax=ax
    )

    # Adicionar os nomes dos youtubers como rótulos no gráfico
    for youtuber, row in df_pca.iterrows():
        ax.text(row['PC1'] + 0.05, row['PC2'], youtuber, fontsize=9)
        
    # Customizar títulos e eixos
    variancia_explicada = pca.explained_variance_ratio_.sum() * 100
    titulo = f"Visualização PCA dos VTMGs ({tipo_analise.title()})"
    ax.set_title(titulo, fontsize=16, pad=20)
    ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} da variância)", fontsize=12)
    ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} da variância)", fontsize=12)
    ax.set_title(f'{titulo}\nVariância Total Explicada pelos 2 Componentes: {variancia_explicada:.2f}%', fontsize=16, pad=20)

    # Salvar o Gráfico
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True)

    if tipo_analise == 'sentimento':
        output_filename = 'pca_visualizacao_sentimento.png'
    else:
        output_filename = f'pca_visualizacao_toxicidade_{n}.png'
    
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"Gráfico PCA salvo em: [green]{output_path}[/green]")

'''
    Função para encontrar as matrizes de transição de todos os vídeos de um youtubers, achatá-las em um vetor e retornar um DataFrame consolidado
    @param base_path - Caminho para o diretório raiz do youtuber (ex: 'files/Tex HS').
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
    @return df_vetores = DataFrame em que cada linha é um vídeo e as colunas são as características achatadas
'''
def criar_vetores_de_caracteristicas_achatados_video(base_path: Path, tipo_analise: str, n: int = None) -> pd.DataFrame:
    vetores_videos = {}

    # Definir o padrão de busca pelo arquivo da matriz
    if tipo_analise == 'sentimento':
        search_pattern = '**/transicoes/matriz_transicao_sentimento.csv'
    else: # toxicidade
        search_pattern = f'**/transicoes/matriz_transicao_{n}.csv'
        
    # Usa .glob() para encontrar todas as matrizes de vídeo individuais
    for matriz_path in base_path.glob(search_pattern):
        try:
            # Identificar a pasta do vídeo
            video_folder = str(matriz_path.parent.parent)

            # Determinar o identificador do vídeo
            video_id = pd.read_csv(f'{video_folder}/videos_info.csv')['video_id'][0]
            
            matriz_df = pd.read_csv(matriz_path, index_col=0)

            vetor_achatado = matriz_df.to_numpy().flatten()

            vetores_videos[video_id] = vetor_achatado
        except Exception as e:
            console.print(f"Inválido (criar_vetores_de_caracteristicas_achatados_video): {e}")

    if not vetores_videos:
        return None

    # Converter para DataFrame e criar os nomes das colunas
    df_vetores = pd.DataFrame.from_dict(vetores_videos, orient='index')

    estados = pd.read_csv(list(base_path.glob(search_pattern))[0], index_col=0).index.tolist()

    nomes_colunas = [f'P_{origem}_para_{destino}' for origem in estados for destino in estados]

    df_vetores.columns = nomes_colunas
    
    return df_vetores

'''
    Função para aplicar o escalonamento e o PCA aos vetores de características dos vídeos de um youtuber e gerar um gráfico de dispersão 2D
    @param df_vetores - DataFrame de vetores de características
    @param youtuber_name - Nome do youtuber para customizar títulos e nomes de arquivos
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
'''
def visualizar_pca_por_video(df_vetores: pd.DataFrame, youtuber_name: str, tipo_analise: str, n: int = None):
    # Testar se o vetor de características é válido
    if df_vetores is None or df_vetores.shape[0] < 2:
        console.print(f"[bold yellow]Aviso: Dados insuficientes para PCA para {youtuber_name} (necessário ao menos 2 vídeos).[/bold yellow]")
        return
        
    # Escalonar os dados com o StandardScaler
    scaler = StandardScaler()
    dados_escalados = scaler.fit_transform(df_vetores)
    
    # Aplicar o PCA
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(dados_escalados)
    df_pca = pd.DataFrame(data=componentes_principais, columns=['PC1', 'PC2'], index=df_vetores.index)
    
    # Gerar o gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    sns.scatterplot(x='PC1', y='PC2', data=df_pca, s=50, alpha=0.8, ax=ax)
        
    # Customizar títulos e eixos
    variancia_explicada = pca.explained_variance_ratio_.sum() * 100
    titulo = f"Visualização PCA dos Vídeos de '{youtuber_name}' ({tipo_analise.title()})"
    ax.set_title(f'{titulo}\nVariância Total Explicada: {variancia_explicada:.2f}%', fontsize=16, pad=20)
    ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)

    # Salvar o Gráfico
    output_dir = Path('files') / youtuber_name / 'transicoes'
    output_dir.mkdir(exist_ok=True)
    
    if tipo_analise == 'sentimento':
        output_filename = f'pca_videos_sentimento.png'
    else:
        output_filename = f'pca_videos_toxicidade_{n}.png'
    
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"Gráfico PCA para [bold cyan]{youtuber_name}[/bold cyan] salvo em: [green]{output_path}[/green]")

'''
    Função para orquestrar a análise de PCA por vídeo para uma lista de youtubers
    @param youtubers_list - Lista de youtubers a serem analisados
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
'''
def analisar_videos_de_youtubers(youtubers_list: list[str], tipo_analise: str, n: int = None):
    console.print(f"\n===== INICIANDO ANÁLISE DE PCA POR VÍDEO ({tipo_analise.upper()}) =====", style="bold blue")

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue
            
        # Criar os vetores de características para todos os vídeos do youtuber
        df_vetores_videos = criar_vetores_de_caracteristicas_achatados_video(base_path, tipo_analise, n)
        
        # Gerar o gráfico PCA com base nesses vetores
        visualizar_pca_por_video(df_vetores_videos, youtuber, tipo_analise, n)

'''
    Função para calcular e plotar o Método do Cotovelo e o Score de Silhueta para uma faixa de k valores
    @param df_vetores - DataFrame de vetores de catacterísticas
    @param max_k - Número máximo de clusters a serem testados (mínimo é 2)
    @param youtuber_name - Nome do youtuber para customização
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
    @return k_otimo - Número ideal de clusters, segundo o Score de Silhueta
'''
def plotar_diagnostico_kmeans(df_vetores: pd.DataFrame, max_k: int, youtuber_name: str, tipo_analise: str, n: int = None) -> int:
    # Testar se o vetor de características é válido
    if df_vetores is None or df_vetores.shape[0] <= max_k:
        console.print(f"[bold yellow]Aviso: Dados insuficientes para diagnóstico k-means para {youtuber_name} (nº de vídeos < max_k).[/bold yellow]")
        return None

    # Escalonar os dados com o StandardScaler
    scaler = StandardScaler()
    dados_escalados = scaler.fit_transform(df_vetores)
    
    k_range = range(2, max_k + 1)
    inercia = []
    silhouette_scores = []

    console.print(f"Testando k de 2 a {max_k} para [bold cyan]{youtuber_name}[/bold cyan]...")

    # Testar todos os valores de k possíveis
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(dados_escalados)
        inercia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(dados_escalados, kmeans.labels_))
    
    # Gerar os gráficos de diagnóstico
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Diagnóstico K-Means para os Vídeos de "{youtuber_name}"', fontsize=18)

    # Gráfico do Método do Cotovelo
    axes[0].plot(k_range, inercia, 'bo-')
    axes[0].set_xlabel('Número de Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inércia (WCSS)', fontsize=12)
    axes[0].set_title('Método do Cotovelo (Elbow Method)', fontsize=14)
    axes[0].grid(True)

    # Gráfico do Score de Silhueta
    axes[1].plot(k_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('Número de Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Score Médio de Silhueta', fontsize=12)
    axes[1].set_title('Análise de Silhueta (Silhouette Analysis)', fontsize=14)
    axes[1].grid(True)
    
    # Salvar o gráfico
    output_dir = Path('files') / youtuber_name / 'transicoes'
    output_dir.mkdir(exist_ok=True)

    if tipo_analise == 'sentimento':
        output_filename = f'kmeans_diagnostico_sentimento.png'
    else:
        output_filename = f'kmeans_diagnostico_toxicidade_{n}.png'

    output_path = output_dir / output_filename

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"Gráfico de diagnóstico salvo em: [green]{output_path}[/green]")
    
    # Retornar o k que deu o maior score de silhueta
    k_otimo = k_range[np.argmax(silhouette_scores)]
    return k_otimo

'''
    Função para aplicar o k-means com um 'k' ótimo e visualizar os clusters em um gráfico PCA bidimensional
    @param df_vetores - DataFrame de vetores de catacterísticas
    @param k_otimo - Número ideal de clusters encontrados
    @param youtuber_name - Nome do youtuber para customização
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
'''
def visualizar_clusters_pca(df_vetores: pd.DataFrame, k_otimo: int, youtuber_name: str, tipo_analise: str, n: int = None):
    # Testar se o vetor de características é válido
    if df_vetores is None or df_vetores.shape[0] < k_otimo:
        return

    # Escalonar os dados com o StandardScaler
    scaler = StandardScaler()
    dados_escalados = scaler.fit_transform(df_vetores)
    
    kmeans = KMeans(n_clusters=k_otimo, n_init='auto', random_state=42)
    clusters = kmeans.fit_predict(dados_escalados)

    # Aplicar o PCA para visualização
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(dados_escalados)
    
    # Criar o DataFrame para plotagem
    df_pca = pd.DataFrame(data=componentes_principais, columns=['PC1', 'PC2'], index=df_vetores.index)
    df_pca['cluster'] = clusters
    
    # Gerar o gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Criar o gráfico de dispersão (scatterplot)
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='cluster',  # Colore os pontos com base na coluna 'cluster'
        palette='viridis', # Paleta de cores
        data=df_pca, 
        s=50,
        alpha=0.9,
        ax=ax,
        legend='full'
    )
        
    # Customizar títulos e eixos
    variancia_explicada = pca.explained_variance_ratio_.sum() * 100
    titulo = f"Clusters de Vídeos (k={k_otimo}) de '{youtuber_name}' ({tipo_analise.title()})"
    ax.set_title(f'{titulo}\nVariância Total Explicada (PCA): {variancia_explicada:.2f}%', fontsize=16, pad=20)
    ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.legend(title='Cluster')
    
    # Salvar o Gráfico
    output_dir = Path('files') / youtuber_name / 'transicoes'
    output_dir.mkdir(exist_ok=True)

    if tipo_analise == 'sentimento':
        output_filename = f'pca_clusters_sentimento_k{k_otimo}.png'
    else:
        output_filename = f'pca_clusters_toxicidade_{n}_k{k_otimo}.png'

    output_path = output_dir / output_filename

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"Gráfico de clusters PCA salvo em: [green]{output_path}[/green]\n")

'''
    Função  para orquestrar a análise de clustering por vídeo para uma lista de youtubers
    @param youtubers_list - Lista de youtubers a serem analisados
    @param tipo_analise - Flag que representa o tipo da análise a ser feita. Deve ser 'sentimento' ou 'toxicidade'
    @param n - Número de estados (necessário apenas se o tipo_analise for 'toxicidade')
    @param max_k - Número máximo de clusters a serem testados (padrão é 8)
'''
def analisar_clusters_de_videos(youtubers_list: list[str], tipo_analise: str, n: int = None, max_k: int = 8):
    console.print(f"\n===== INICIANDO ANÁLISE DE CLUSTERING POR VÍDEO ({tipo_analise.upper()}) =====", style="bold blue")

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue
            
        # Criar os vetores de características para todos os vídeos do youtuber
        df_vetores_videos = criar_vetores_de_caracteristicas_achatados_video(base_path, tipo_analise, n)
        
        if df_vetores_videos is None:
            continue

        # Gerar os gráficos de diagnóstico e encontrar o k ótimo
        k_otimo = plotar_diagnostico_kmeans(df_vetores_videos, max_k, youtuber, tipo_analise, n)
        
        if k_otimo is None:
            continue
            
        console.print(f"O número ideal de clusters para [bold cyan]{youtuber}[/bold cyan] (via Silhueta) é: [bold magenta]{k_otimo}[/bold magenta]")

        # Gerar o gráfico PCA com os vídeos coloridos pelos clusters
        visualizar_clusters_pca(df_vetores_videos, k_otimo, youtuber, tipo_analise, n)

if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    #salvar_transicoes_sentimento(lista_youtubers)

    #salvar_matriz_transicao_sentimento_video(lista_youtubers)

    #salvar_matriz_transicao_sentimento_youtuber(lista_youtubers)

    #analisar_distribuicao_estacionaria_sentimento(lista_youtubers)

    #tranalisar_tempo_permanencia_sentimento(lista_youtubers)

    #df_vetores_sentimento = criar_vetores_de_caracteristicas_achatados_youtuber(lista_youtubers, 'sentimento')
    #visualizar_pca_youtuber(df_vetores_sentimento, 'sentimento')

    #analisar_videos_de_youtubers(lista_youtubers, 'sentimento')

    analisar_clusters_de_videos(lista_youtubers, 'sentimento')
