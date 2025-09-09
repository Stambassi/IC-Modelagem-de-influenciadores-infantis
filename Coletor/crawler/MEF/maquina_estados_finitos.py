from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

'''
    Função para agregar os scores de toxicidade de todos os vídeos da lista de youtubers, gerar um histograma consolidado e salvá-los como uma imagem
    @param youtubers_list - Lista de youtubers a serem analisados
    @param bins - Número de barras (divisões) a serem usadas no histograma (padrão é 100)
'''
def gerar_histograma_toxicidade(youtubers_list: list[str], bins: int = 100) -> None:
    console.print(f"\n--- Geração de Histograma de Toxicidade (Agregado) ---", style="bold magenta")

    # Lista para armazenar todos os scores de toxicidade de todos os vídeos
    todas_as_toxicidades = []

    # Coletar os dados
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue

        console.print(f'Coletando dados de [bold cyan]{youtuber}[/bold cyan]...')
        
        for tiras_csv_path in base_path.rglob('tiras_video.csv'):
            try:
                df_video = pd.read_csv(tiras_csv_path)
                if not df_video.empty and 'toxicidade' in df_video.columns:
                    # Adiciona os scores da coluna 'toxicidade' à nossa lista geral
                    todas_as_toxicidades.extend(df_video['toxicidade'].tolist())
            except Exception as e:
                console.print(f"Inválido (gerar_histograma_toxicidade): {e}")

    if not todas_as_toxicidades:
        console.print("[bold yellow]Aviso: Nenhum dado de toxicidade foi encontrado para gerar o histograma.[/bold yellow]")
        return

    console.print(f"\n[green]Dados coletados![/green] Total de {len(todas_as_toxicidades)} segmentos de 1 minuto analisados.")
    console.print("Gerando o gráfico de histograma...")

    # Cria uma figura e um conjunto de eixos para o gráfico
    fig, ax = plt.subplots(figsize=(12, 7))

    # Comando principal para criar o histograma
    ax.hist(todas_as_toxicidades, bins=bins, color='#007acc', edgecolor='black', alpha=0.7)

    # Customizar o gráfico para clareza
    ax.set_title('Distribuição Agregada dos Scores de Toxicidade', fontsize=16, pad=20)
    ax.set_xlabel('Score de Toxicidade (de 0.0 a 1.0)', fontsize=12)
    ax.set_ylabel('Frequência (Nº de Segmentos de 1 min)', fontsize=12)
    
    # IMPORTANTE: Usar escala logarítmica no eixo Y
    # Isso permite visualizar a frequência de scores altos, que de outra forma seriam achatados no eixo
    #ax.set_yscale('log')
    
    # Garantir que o eixo X vá exatamente de 0 a 1
    ax.set_xlim(0, 1) 

    # Adicionar uma grade horizontal para facilitar a leitura
    ax.grid(axis='y', linestyle='--', alpha=0.6) 
    
    # Salvar o gráfico em um arquivo
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'histograma_toxicidade_agregado_{bins}.png'

    # Salvar a figura. dpi=300 para alta resolução, bbox_inches='tight' para não cortar os títulos
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"\nHistograma salvo com sucesso em: [green]{output_path}[/green]")

'''
    Função para armazenar as transições de estados de cada vídeo em um arquivo CSV
    @param youtubers_list - Lista de youtubers a serem analisados
    @param n - quantidade de subgrupos de toxicidade a serem divididos
'''
def salvar_transicoes(youtubers_list: list[str], n: int = 4) -> None:
    # Testar quantidade de subgrupos
    if not n > 0:
        console.print('Inválido (salvar_transicoes): Número de subgrupos não é permitido.')

    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar um objeto Path para o diretório base do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se o diretório do youtuber não existir, pular para o próximo
        if not base_path.is_dir():
            continue

        print(f'>>> Processando {youtuber}')

        # .rglob('tiras_video.csv') busca recursivamente por este arquivo em todas as subpastas.
        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            # O diretório do vídeo é a pasta "pai" do arquivo CSV encontrado
            video_path = tiras_csv_path.parent
            
            try:
                # Criar a pasta 'transicoes' dentro da pasta do vídeo, se não existir
                (video_path / 'transicoes').mkdir(parents=True, exist_ok=True)
                
                # Ler o arquivo CSV que já foi encontrado
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se há dados suficientes para uma transição
                if df_tiras_video.empty or 'toxicidade' not in df_tiras_video.columns or len(df_tiras_video) < 2:
                    continue

                # Criar 'n + 1' pontos de corte entre 0 e 1. Ex: n = 4 -> [0.0, 0.25, 0.5, 0.75, 1.0]
                grupos = np.linspace(0, 1, n + 1)

                # Criar as etiquetas para os estados. Ex: n = 4 -> [1, 2, 3, 4]
                labels = range(1, n + 1)

                # Usar pd.cut para classificar cada valor de 'toxicidade' em um estado
                # 'include_lowest=True' garante que o valor 0.0 seja incluído no primeiro estado
                df_tiras_video['estado'] = pd.cut(
                    df_tiras_video['toxicidade'],
                    bins=grupos,
                    labels=labels,
                    include_lowest=True
                )

                # Criar uma nova coluna 'proximo_estado' que contém o estado da linha seguinte
                # O método .shift(-1) "puxa" a coluna uma linha para cima
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)

                # Remover a última linha, pois ela não tem um 'proximo_estado' (resulta em NaN)
                # Também remover linhas caso haja algum outro valor NaN para garantir a integridade dos dados
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa por estado de origem ('estado') e estado de destino ('proximo_estado') e conta o número de ocorrências em cada grupo.
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')

                # Ordenar para uma visualização mais clara no CSV
                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Define o nome do arquivo de saída para que ele inclua o número de estados 'n'
                output_path = video_path / 'transicoes' / f'transicoes_{n}.csv'
                
                # Salva o DataFrame com as contagens no arquivo CSV, sem o índice do pandas
                contagem.to_csv(output_path, index=False)

                console.print(f"Arquivo CSV gerado: {output_path}")
            
            except Exception as e:
                console.print(f'Inválido (salvar_transicoes): {e}')


'''
    Função para criar e persistir a Matriz de Transição da Cadeia de Markov para cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
    @param n - quantidade de subgrupos de toxicidade a serem divididos
'''
def salvar_matriz_transicao_video(youtubers_list: list[str], n: int = 4) -> None:
    # Testar quantidade de subgrupos
    if not n > 0:
        console.print('Inválido (salvar_matriz_transicao_video): Número de subgrupos não é permitido.')

    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar um objeto Path para o diretório base do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se o diretório do youtuber não existir, pular para o próximo
        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes (n={n}) para [bold cyan]{youtuber}[/bold cyan]')

        # .rglob(f'transicoes_{n}.csv') busca recursivamente por este arquivo em todas as subpastas.
        for transicoes_csv_path in base_path.rglob(f'transicoes_{n}.csv'):                
            try:                
                # Ler o arquivo CSV que já foi encontrado
                df_transicoes = pd.read_csv(transicoes_csv_path)

                # Se o arquivo de contagens estiver vazio, pular para o próximo
                if df_transicoes.empty:
                    continue

                # Calcular a soma das transições que saem de cada estado
                # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence.
                somas_por_estado = df_transicoes.groupby('estado')['contagem'].transform('sum')

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
                
                # O arquivo da matriz será salvo na mesma pasta 'transicoes'
                output_path = transicoes_csv_path.parent / f'matriz_transicao_{n}.csv'
                matriz_transicao.to_csv(output_path)
                console.print(f"Matriz de Transição de vídeo gerada: {output_path}")

            except Exception as e:
                console.print(f'Inválido (salvar_matriz_transicao_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição agregada para cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param n - quantidade de subgrupos de toxicidade a serem divididos
'''
def salvar_matriz_transicao_youtubers(youtubers_list: list[str], n: int = 4) -> None:
    # Testar quantidade de subgrupos
    if not n > 0:
        console.print('Inválido (salvar_matriz_transicao_youtubers): Número de subgrupos não é permitido.')

    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matriz agregada (n={n}) para [bold cyan]{youtuber}[/bold cyan]')

        try:
            # Encontrar e concatenar todas as transições do youtuber
            lista_dfs_transicoes = []

            # .rglob(f'transicoes_{n}.csv') busca recursivamente por este arquivo.
            for transicoes_csv_path in base_path.rglob(f'transicoes_{n}.csv'):
                # Adicionar o DataFrame de cada vídeo a uma lista
                df_video = pd.read_csv(transicoes_csv_path)

                # Testar se o resultado não está vazio
                if not df_video.empty:
                    lista_dfs_transicoes.append(df_video)
            
            # Se nenhum arquivo de transição foi encontrado para este youtuber, pular para o próximo
            if not lista_dfs_transicoes:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições encontrado para {youtuber}. Pulando.[/yellow]")
                continue
            
            # Concatena todos os DataFrames da lista em um só
            df_agregado = pd.concat(lista_dfs_transicoes, ignore_index=True)

            # Agrupar por cada tipo de transição e somar as contagens de todos os vídeos
            df_soma_total = df_agregado.groupby(['estado', 'proximo_estado'])['contagem'].sum().reset_index()

            # Calcular a soma das transições que saem de cada estado
            # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence.
            somas_por_estado = df_soma_total.groupby('estado')['contagem'].transform('sum')

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
            
            # Criar a pastas 'transicoes' dentro da pasta de cada youtuber, se não existir
            (base_path / 'transicoes').mkdir(parents=True, exist_ok=True)

            # O arquivo é salvo na pasta raiz do youtuber
            output_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{n}.csv'
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição de youtuber gerada: {output_path}")

        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_youtubers): {e}')

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
    Função para carregar a matriz de transição de cada youtuber, calcular sua distribuição estacionária e exibir os resultados
    @param youtubers_list - Lista de youtubers a serem analisados
    @param n - quantidade de subgrupos de toxicidade a serem divididos
'''
def analisar_distribuicao_estacionaria_youtubers(youtubers_list: list[str], n: int = 4) -> None:
    console.print(f"\n--- Análise da Distribuição Estacionária (n={n}) ---", style="bold magenta")
    
    # Dicionário para armazenar os resultados
    resultados = {}

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{n}.csv'

        if not matriz_path.exists():
            continue

        try:
            # Carregar a matriz, definindo a primeira coluna como índice
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            
            # Calcular a distribuição usando a função auxiliar
            distribuicao = calcular_distribuicao_estacionaria(matriz_df)
            
            # Armazenar o resultado
            resultados[youtuber] = distribuicao

        except Exception as e:
            console.print(f"Inválido (analisar_distribuicao_estacionaria_youtubers): {e}")
    
    # Exibir os resultados de forma organizada
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Criar um DataFrame com os resultados para fácil visualização e comparação
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    df_resultados.columns = [f'Estado {i+1}' for i in range(n)]
    
    # Ordenar pelo estado de maior toxicidade para identificar os youtubers mais problemáticos
    df_resultados = df_resultados.sort_values(by=f'Estado {n}', ascending=False)

    # Definir o caminho para o arquivo de saída
    output_dir = Path('files/transicoes')

    # Garantir que o diretório 'files/transicoes' exista
    output_dir.mkdir(exist_ok=True) 

    output_path = output_dir / f'distribuicao_estacionaria_geral_{n}.csv'

    # Salvar o DataFrame em um arquivo CSV.
    df_resultados.to_csv(output_path, index=True, index_label='youtuber')

    console.print(f"\nResultados da distribuição estacionária salvos em: [green]{output_path}[/green]")

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
    Função para ler a matriz de transição de cada youtuber, calcular o tempo médio de permanência em cada estado, exibir os resultados 
    e salvá-los em um arquivo CSV.
    @param youtubers_list - Lista de youtubers a serem analisados
    @param n - quantidade de subgrupos de toxicidade a serem divididos
'''
def analisar_tempo_permanencia_youtubers(youtubers_list: list[str], n: int = 4) -> None:
    console.print(f"\n--- Análise do Tempo Médio de Permanência (n={n}) ---", style="bold magenta")
    
    resultados = {}

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{n}.csv'

        if not matriz_path.exists():
            continue

        try:
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            tempos = calcular_tempo_permanencia(matriz_df)
            resultados[youtuber] = tempos
        except Exception as e:
            console.print(f"Inválido (analisar_tempo_permanencia_youtubers): {e}")
    
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Converter dicionário em DataFrame
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    
    # Renomeia as colunas para maior clareza
    df_resultados.columns = [f'Estado {i}' for i in df_resultados.columns]
    
    # Ordenar pelo tempo de permanência no estado de maior toxicidade
    df_resultados = df_resultados.sort_values(by=f'Estado {n}', ascending=False)
    
    # Persistir os dados
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'tempo_permanencia_geral_{n}.csv'
    df_resultados.to_csv(output_path, index=True, index_label='youtuber')

    console.print(f"\nResultados do tempo de permanência salvos em: [green]{output_path}[/green]")

if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    #gerar_histograma_toxicidade(lista_youtubers)

    salvar_transicoes(lista_youtubers, 3)
    #salvar_transicoes(lista_youtubers, 4)
    #salvar_transicoes(lista_youtubers, 5)

    salvar_matriz_transicao_video(lista_youtubers, 3)
    #salvar_matriz_transicao_video(lista_youtubers, 4)
    #salvar_matriz_transicao_video(lista_youtubers, 5)

    salvar_matriz_transicao_youtubers(lista_youtubers, 3)
    #salvar_matriz_transicao_youtubers(lista_youtubers, 4)
    #salvar_matriz_transicao_youtubers(lista_youtubers, 5)

    analisar_distribuicao_estacionaria_youtubers(lista_youtubers, 3)
    #analisar_distribuicao_estacionaria_youtubers(lista_youtubers, 4)
    #analisar_distribuicao_estacionaria_youtubers(lista_youtubers, 5)

    analisar_tempo_permanencia_youtubers(lista_youtubers, 3)
    #analisar_tempo_permanencia_youtubers(lista_youtubers, 4)
    #analisar_tempo_permanencia_youtubers(lista_youtubers, 5)
