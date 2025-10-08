from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

'''
    Função para encontrar e apagar arquivos de análise legados (sem métrica no nome) para evitar conflitos
    @param youtubers_list - Lista de youtubers para limpar as pastas.
    @param n_list - Lista de valores de 'n' usados em análises antigas (ex: [3, 4, 5]).
'''
def limpar_arquivos_legado(youtubers_list: list[str], n_list: list[int]) -> None:
    console.print(f"\n--- [bold yellow]Buscando por arquivos de análise legados para limpeza[/bold yellow] ---")
    files_to_delete = []

    # Buscar por arquivos globais antigos
    for n in n_list:
        files_to_delete.extend(Path('files/transicoes').glob(f'distribuicao_estacionaria_geral_{n}.csv'))
        files_to_delete.extend(Path('files/transicoes').glob(f'tempo_permanencia_geral_{n}.csv'))
    files_to_delete.extend(Path('files/transicoes').glob('histograma_toxicidade_*.png'))

    # Buscar por arquivos de vídeo antigos
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        for n in n_list:
            files_to_delete.extend(base_path.rglob(f'transicoes_{n}.csv'))
            files_to_delete.extend(base_path.rglob(f'matriz_transicao_{n}.csv'))
    
    if not files_to_delete:
        console.print("Nenhum arquivo legado encontrado.")
        return

    console.print("Os seguintes arquivos legados (sem métrica no nome) foram encontrados:")
    for f in sorted(files_to_delete):
        console.print(f"- [dim]{f}[/dim]")
    
    confirm = console.input("\nVocê confirma a exclusão permanente de todos esses arquivos? (s/N): ").lower()
    if confirm == 's':
        deleted_count = 0
        for f in files_to_delete:
            try:
                f.unlink()
                deleted_count += 1
            except OSError:
                pass # Ignora se o arquivo já foi deletado
        console.print(f"\n[green]{deleted_count} arquivos legados foram removidos.[/green]")
    else:
        console.print("\nLimpeza cancelada.")


'''
    Função para agregar os scores de uma métrica de todos os vídeos da lista de youtubers, gerar um histograma consolidado e salvá-los como uma imagem
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A coluna do DataFrame a ser analisada (ex: 'negatividade', 'toxicidade')
    @param bins - Número de barras (divisões) a serem usadas no histograma (padrão é 100)
'''
def gerar_histograma(youtubers_list: list[str], metrica: str, bins: int = 100) -> None:
    console.print(f"\n--- Geração de Histograma de {metrica.capitalize()} (Agregado) ---", style="bold magenta")

    todos_os_scores = []

    # Coletar os dados
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue

        console.print(f'Coletando dados de [bold cyan]{youtuber}[/bold cyan]...')
        
        for tiras_csv_path in base_path.rglob('tiras_video.csv'):
            try:
                df_video = pd.read_csv(tiras_csv_path)

                # Verificar pela métrica fornecida
                if not df_video.empty and metrica in df_video.columns:
                    todos_os_scores.extend(df_video[metrica].tolist())
            except Exception as e:
                console.print(f"Inválido (gerar_histograma): {e}")

    if not todos_os_scores:
        console.print(f"[bold yellow]Aviso: Nenhum dado de '{metrica}' foi encontrado para gerar o histograma.[/bold yellow]")
        return

    console.print(f"\n[green]Dados coletados![/green] Total de {len(todos_os_scores)} segmentos analisados.")
    console.print("Gerando o gráfico de histograma...")

    # Criar uma figura e um conjunto de eixos para o gráfico
    fig, ax = plt.subplots(figsize=(12, 7))

    # Comando principal para criar o histograma
    ax.hist(todos_os_scores, bins=bins, color='#007acc', edgecolor='black', alpha=0.7)

    # Customizar o gráfico para clareza
    ax.set_title(f'Distribuição Agregada dos Scores de {metrica.capitalize()}', fontsize=16, pad=20)
    ax.set_xlabel(f'Score de {metrica.capitalize()} (de 0.0 a 1.0)', fontsize=12)
    ax.set_ylabel('Frequência (Nº de Segmentos)', fontsize=12)
    
    # Usar escala logarítmica no eixo Y
    # Isso permite visualizar a frequência de scores altos, que de outra forma seriam achatados no eixo
    # ax.set_yscale('log')
    
    # Garantir que o eixo X vá exatamente de 0 a 1
    ax.set_xlim(0, 1) 

    # Adicionar uma grade horizontal para facilitar a leitura
    ax.grid(axis='y', linestyle='--', alpha=0.6) 
    
    # Salvar o gráfico em um arquivo
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'histograma_{metrica}_agregado_{bins}.png'

    # Salvar a figura. dpi=300 para alta resolução, bbox_inches='tight' para não cortar os títulos
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"\nHistograma salvo com sucesso em: [green]{output_path}[/green]")


'''
    Função para armazenar as transições de estados de cada vídeo em um arquivo CSV
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A coluna do DataFrame a ser discretizada em estados
    @param n - quantidade de subgrupos a serem divididos
'''
def salvar_transicoes(youtubers_list: list[str], metrica: str, n: int = 4) -> None:
    # Testar quantidade de subgrupos
    if not n > 0:
        console.print('Inválido (salvar_transicoes): Número de subgrupos não é permitido.')
        return 

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
                if df_tiras_video.empty or metrica not in df_tiras_video.columns or len(df_tiras_video) < 2:
                    continue

                # Criar 'n + 1' pontos de corte entre 0 e 1. Ex: n = 4 -> [0.0, 0.25, 0.5, 0.75, 1.0]
                grupos = np.linspace(0, 1, n + 1)

                # Criar as etiquetas para os estados. Ex: n = 4 -> [1, 2, 3, 4]
                labels = range(1, n + 1)

                # Usar pd.cut para classificar cada valor da métrica em um estado
                # 'include_lowest=True' garante que o valor 0.0 seja incluído no primeiro estado
                df_tiras_video['estado'] = pd.cut(
                    df_tiras_video[metrica],
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

                # Define o nome do arquivo de saída para que ele inclua a métrica e o número de estados 'n'
                output_path = video_path / 'transicoes' / f'transicoes_{metrica}_{n}.csv'
                
                # Salva o DataFrame com as contagens no arquivo CSV, sem o índice do pandas
                contagem.to_csv(output_path, index=False)

            except Exception as e:
                console.print(f'Inválido (salvar_transicoes): {e}')


'''
    Função para criar e persistir a Matriz de Transição da Cadeia de Markov para cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A métrica que está sendo analisada
    @param n - quantidade de subgrupos a serem divididos
'''
def salvar_matriz_transicao_video(youtubers_list: list[str], metrica: str, n: int = 4) -> None:
    # Testar quantidade de subgrupos
    if not n > 0:
        console.print('Inválido (salvar_matriz_transicao_video): Número de subgrupos não é permitido.')
        return

    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar um objeto Path para o diretório base do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se o diretório do youtuber não existir, pular para o próximo
        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes de "{metrica}" (n={n}) para [bold cyan]{youtuber}[/bold cyan]')

        # .rglob busca recursivamente por este arquivo em todas as subpastas.
        for transicoes_csv_path in base_path.rglob(f'transicoes_{metrica}_{n}.csv'):                
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
                output_path = transicoes_csv_path.parent / f'matriz_transicao_{metrica}_{n}.csv'
                matriz_transicao.to_csv(output_path)

            except Exception as e:
                console.print(f'Inválido (salvar_matriz_transicao_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição agregada para cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A métrica que está sendo analisada
    @param n - quantidade de subgrupos a serem divididos
'''
def salvar_matriz_transicao_youtubers(youtubers_list: list[str], metrica: str, n: int = 4) -> None:
    # Testar se o número de subgrupos é válido
    if not n > 0:
        console.print('Inválido (salvar_matriz_transicao_youtubers): Número de subgrupos não é permitido.')
        return

    # Iterar sobre cada youtuber da lista
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir():
            continue
        console.print(f'>>> Processando matriz agregada de "{metrica}" (n={n}) para [bold cyan]{youtuber}[/bold cyan]')
        try:
            # Lista para armazenar os DataFrames de transição de cada vídeo
            lista_dfs_transicoes = []
            
            # Encontrar e carregar todos os arquivos de transições do youtuber
            for transicoes_csv_path in base_path.rglob(f'transicoes_{metrica}_{n}.csv'):
                df_video = pd.read_csv(transicoes_csv_path)
                if not df_video.empty:
                    lista_dfs_transicoes.append(df_video)
            
            # Se nenhum arquivo for encontrado, pular para o próximo youtuber
            if not lista_dfs_transicoes:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições de '{metrica}' encontrado para {youtuber}.[/yellow]")
                continue
            
            # Unir os dados de todos os vídeos em um único DataFrame
            df_agregado = pd.concat(lista_dfs_transicoes, ignore_index=True)
            
            # Agrupar por tipo de transição e somar as contagens de todos os vídeos
            df_soma_total = df_agregado.groupby(['estado', 'proximo_estado'])['contagem'].sum().reset_index()
            
            # Calcular a soma total de saídas de cada estado
            somas_por_estado = df_soma_total.groupby('estado')['contagem'].transform('sum')
            
            # Calcular a probabilidade de cada transição (contagem da transição / total de saídas do estado)
            probabilidade = df_soma_total['contagem'] / somas_por_estado
            df_soma_total['probabilidade'] = probabilidade.fillna(0)
            
            # Reorganizar (pivotar) o DataFrame para o formato de matriz de transição
            matriz_transicao_youtuber = df_soma_total.pivot(index='estado', columns='proximo_estado', values='probabilidade')
            
            # Garantir que a pasta de saída exista
            (base_path / 'transicoes').mkdir(parents=True, exist_ok=True)
            
            # Definir o caminho e salvar a matriz agregada em um arquivo CSV
            output_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{metrica}_{n}.csv'
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição de youtuber gerada: {output_path}")
        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_youtubers): {e}')

'''
    Função para calcular a distribuição estacionária de uma Cadeia de Markov a partir de sua matriz de transição
    @param matriz_transicao - Matriz de transição n x n, em que as linhas somam 1
    @return distribuicao_estacionaria - Vetor de probabilidades com a distribuição estacionária 'pi' de um youtuber 
'''
def calcular_distribuicao_estacionaria(matriz_transicao: pd.DataFrame) -> np.ndarray:
    # Converter o DataFrame para uma matriz NumPy para cálculos de álgebra linear
    matriz_np = matriz_transicao.to_numpy()

    # Calcular autovalores e autovetores da matriz transposta (para encontrar o autovetor à esquerda)
    autovalores, autovetores = np.linalg.eig(matriz_np.T)
    
    try:
        # Encontrar o índice do autovalor que é numericamente próximo de 1
        indice_estacionario = np.where(np.isclose(autovalores, 1))[0][0]
        
        # Selecionar o autovetor correspondente a esse autovalor
        vetor_estacionario = autovetores[:, indice_estacionario]

        # Extrair apenas a parte real do autovetor (a parte imaginária deve ser próxima de zero)
        vetor_estacionario_real = np.real(vetor_estacionario)

        # Normalizar o vetor para que a soma de suas componentes seja 1, resultando em um vetor de probabilidade
        distribuicao_estacionaria = vetor_estacionario_real / np.sum(vetor_estacionario_real)
        
        return distribuicao_estacionaria
    except IndexError: # Lida com casos onde nenhum autovalor é próximo de 1
        console.print("[yellow]Aviso: Não foi possível encontrar um estado estacionário (autovalor 1). Retornando distribuição uniforme.[/yellow]")
        # Retorna uma distribuição uniforme como fallback em caso de problemas numéricos
        return np.ones(matriz_np.shape[0]) / matriz_np.shape[0]

'''
    Função para carregar a matriz de transição de cada youtuber, calcular sua distribuição estacionária e exibir os resultados
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A métrica que está sendo analisada
    @param n - quantidade de subgrupos a serem divididos
'''
def analisar_distribuicao_estacionaria_youtubers(youtubers_list: list[str], metrica: str, n: int = 4) -> None:
    console.print(f"\n--- Análise da Distribuição Estacionária de '{metrica}' (n={n}) ---", style="bold magenta")
    
    # Dicionário para armazenar os resultados de cada youtuber
    resultados = {}

    # Iterar sobre cada youtuber para processar seus dados
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{metrica}_{n}.csv'

        if not matriz_path.exists():
            continue
        
        try:
            # Carregar a matriz de transição do arquivo, usando a primeira coluna como índice
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            
            # Calcular a distribuição estacionária usando a função auxiliar
            distribuicao = calcular_distribuicao_estacionaria(matriz_df)
            
            # Armazenar o vetor de distribuição no dicionário de resultados
            resultados[youtuber] = distribuicao
        except Exception as e:
            console.print(f"Inválido (analisar_distribuicao_estacionaria_youtubers): {e}")
    
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Criar um DataFrame a partir do dicionário de resultados para fácil visualização
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    df_resultados.columns = [f'Estado {i+1}' for i in range(n)]
    
    # Ordenar os youtubers pelo estado de maior score (estado n)
    df_resultados = df_resultados.sort_values(by=f'Estado {n}', ascending=False)
    
    # Definir e garantir que o diretório de saída exista
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True, parents=True) 
    output_path = output_dir / f'distribuicao_estacionaria_geral_{metrica}_{n}.csv'

    # Salvar os resultados consolidados em um arquivo CSV
    df_resultados.to_csv(output_path, index=True, index_label='youtuber')
    console.print(f"\nResultados da distribuição estacionária salvos em: [green]{output_path}[/green]")

'''
    Função para calcular o Tempo Médio de Permanência (Sojourn Time) para cada estado
    @param matriz_transicao - Matriz de transição n x n, em que as linhas somam 1
    @return resultado_series - Série com o tempo médio de permanência para cada estado
'''
def calcular_tempo_permanencia(matriz_transicao: pd.DataFrame) -> pd.Series:
    # Extrair os valores da diagonal principal (probabilidade P_ii de permanecer no mesmo estado)
    p_ii = np.diag(matriz_transicao.fillna(0))
    
    # Calcular o tempo de permanência usando a fórmula 1 / (1 - P_ii)
    # Suprimir avisos de divisão por zero caso P_ii seja 1 (estado absorvente)
    with np.errstate(divide='ignore', invalid='ignore'):
        tempos_permanencia = 1 / (1 - p_ii)
    
    # Criar uma Série pandas para um resultado mais legível, com os nomes dos estados como índice
    resultado_series = pd.Series(tempos_permanencia, index=matriz_transicao.index, name='Tempo de Permanencia')
    
    return resultado_series

'''
    Função para ler a matriz de transição de cada youtuber, calcular o tempo médio de permanência em cada estado, exibir os resultados 
    e salvá-los em um arquivo CSV.
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica - A métrica que está sendo analisada
    @param n - quantidade de subgrupos a serem divididos
'''
def analisar_tempo_permanencia_youtubers(youtubers_list: list[str], metrica: str, n: int = 4) -> None:
    console.print(f"\n--- Análise do Tempo Médio de Permanência de '{metrica}' (n={n}) ---", style="bold magenta")
    
    # Dicionário para armazenar os resultados de cada youtuber
    resultados = {}

    # Iterar sobre cada youtuber para processar seus dados
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        matriz_path = base_path / 'transicoes' / f'matriz_transicao_youtuber_{metrica}_{n}.csv'

        if not matriz_path.exists():
            continue

        try:
            # Carregar a matriz de transição do arquivo
            matriz_df = pd.read_csv(matriz_path, index_col=0)
            
            # Calcular o tempo de permanência usando a função auxiliar
            tempos = calcular_tempo_permanencia(matriz_df)
            
            # Armazenar a série de tempos no dicionário de resultados
            resultados[youtuber] = tempos
        except Exception as e:
            console.print(f"Inválido (analisar_tempo_permanencia_youtubers): {e}")
    
    if not resultados:
        console.print("[yellow]Nenhum dado para analisar foi encontrado.[/yellow]")
        return

    # Converter o dicionário de resultados em um DataFrame
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    
    # Renomear as colunas para maior clareza
    df_resultados.columns = [f'Estado {i}' for i in df_resultados.columns]
    
    # Ordenar os youtubers pelo tempo de permanência no estado de maior score (estado n)
    df_resultados = df_resultados.sort_values(by=f'Estado {n}', ascending=False)
    
    # Garantir que o diretório de saída exista e salvar os dados
    output_dir = Path('files/transicoes')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'tempo_permanencia_geral_{metrica}_{n}.csv'
    df_resultados.to_csv(output_path, index=True, index_label='youtuber')

    console.print(f"\nResultados do tempo de permanência salvos em: [green]{output_path}[/green]")

if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']
    metrica = 'negatividade'
    # metrica = 'toxicity'

    #gerar_histograma(lista_youtubers, metrica=metrica)

    salvar_transicoes(lista_youtubers, metrica, 3)
    #salvar_transicoes(lista_youtubers, metrica, 4)
    #salvar_transicoes(lista_youtubers, metrica, 5)

    salvar_matriz_transicao_video(lista_youtubers, metrica, 3)
    #salvar_matriz_transicao_video(lista_youtubers, metrica, 4)
    #salvar_matriz_transicao_video(lista_youtubers, metrica, 5)

    salvar_matriz_transicao_youtubers(lista_youtubers, metrica, 3)
    #salvar_matriz_transicao_youtubers(lista_youtubers, metrica, 4)
    #salvar_matriz_transicao_youtubers(lista_youtubers, metrica, 5)

    analisar_distribuicao_estacionaria_youtubers(lista_youtubers, metrica, 3)
    #analisar_distribuicao_estacionaria_youtubers(lista_youtubers, metrica, 4)
    #analisar_distribuicao_estacionaria_youtubers(lista_youtubers, metrica, 5)

    analisar_tempo_permanencia_youtubers(lista_youtubers, metrica, 3)
    #analisar_tempo_permanencia_youtubers(lista_youtubers, metrica, 4)
    #analisar_tempo_permanencia_youtubers(lista_youtubers, metrica, 5)