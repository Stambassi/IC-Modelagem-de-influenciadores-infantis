import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

def count_folders_os_walk(path):
    """Counts the total number of folders within a given path, including subfolders."""
    folder_count = 0
    for root, dirs, files in os.walk(path):
        folder_count += len(dirs)
    return folder_count

'''
    Função para gerar o titas_total.csv para cada youtuber, contendo a informação de todas as tiras de cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def salvar_tiras_total_youtubers(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            youtuber_data = pd.DataFrame()
            console.rule(youtuber)
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            total_month_videos = count_folders_os_walk(next_month_dir)
                            atual = 1
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                console.print(f"Videos {month_folder}/{year_folder}: {atual}/{total_month_videos} ")
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):                              
                                    # Tentar abrir o arquivo csv
                                    try:
                                        df_tiras = pd.read_csv(f'{next_video_dir}/tiras_video.csv')
                                        youtuber_data = pd.concat([youtuber_data, df_tiras],ignore_index=True)
                                    except Exception as e:
                                        # print(f'Inválido: {next_video_dir}')
                                        pass
                                atual += 1
            if(not youtuber_data.empty):
                tiras_total_path = os.path.join(base_dir, 'tiras_total.csv')
                youtuber_data.to_csv(tiras_total_path, index=False)                     

def gerar_graficos_tiras_media_dp(youtubers_list: list[str]) -> None:
    all_data = pd.DataFrame()
    for youtuber in youtubers_list:
        try:
            base_dir = f"files/{youtuber}"
            tiras_total_path = os.path.join(base_dir, 'tiras_total.csv')
            youtuber_data = pd.read_csv(tiras_total_path)
            
            if(not youtuber_data.empty):
                console.print(f"Gerando dados do [cyan]{youtuber}")
        
                construir_grafico_toxicidade_media_dp(youtuber_data,youtuber,limite_index=40)

                all_data = pd.concat([all_data, youtuber_data],ignore_index=True)
        except Exception as e:
            console.print(f"{youtuber} não possui csv válido")
    if (not all_data.empty):
        console.print(f"Gerando gráfico geral")
        construir_grafico_toxicidade_media_dp(all_data,"",limite_index=40)

def construir_grafico_toxicidade_media_dp(youtuber_data,youtuber,limite_index):
    base_dir = "files"
    youtuber_tiras_total = youtuber_data.groupby('index')['toxicidade'].mean().reset_index()
    youtuber_tiras_total.columns = ['index', f'media toxicidade']   

    youtuber_tiras_filtrada = youtuber_tiras_total[youtuber_tiras_total['index'] < limite_index]
    
    youtuber_tiras_desvio = youtuber_data.groupby('index')['toxicidade'].std().reset_index()
    youtuber_tiras_desvio.columns = ['index', f'desvio padrao']   

    youtuber_tiras_filtrada_desvio = youtuber_tiras_desvio[youtuber_tiras_desvio['index'] < limite_index]

    # Gerar gráfico de toxicidade média ao longo do vídeo
    toxicidade = youtuber_tiras_filtrada['media toxicidade']
    desvio_padrao = youtuber_tiras_filtrada_desvio['desvio padrao']
    indexes = list(range(0, len(toxicidade)))
    plt.plot(indexes, toxicidade, label='Toxicidade no Vídeo')
    plt.plot(indexes, desvio_padrao, label='Desvio Padrão Toxicidade')
    plt.xlabel('Tiras')
    plt.ylabel('Toxicidade')
    plt.ylim(0.0,1.0)
    if len(youtuber) > 1:
        base_dir = f"files/{youtuber}"
        plt.title(f'Análise da Oscilação de Toxicidade média do {youtuber}')
    else:
        plt.title(f'Análise de Oscilação de Toxicidade média')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{base_dir}/grafico_toxicidade_medio_video.png')
    plt.close() 

    console.print("Gráfico [green]salvo")

'''
    Função para gerar os gráficos da média, do desvio padrão e da mediana de toxicidade de um youtuber
    @param youtuber
'''
def estatisticas_youtuber(youtuber: str) -> None:
    # Definições iniciais
    base_dir = f"files/{youtuber}"
    lista_indexes = []
    index = 1
    lista_media = []
    lista_desvio_padrao = []
    lista_mediana = []

    # Percorrer as pastas
    if os.path.isdir(base_dir):
        console.rule(base_dir)
        # Percorrer os anos
        for year_folder in os.listdir(base_dir):
            next_year_dir = os.path.join(base_dir, year_folder)
            if os.path.isdir(next_year_dir):
                # Percorrer os meses
                for month_folder in os.listdir(next_year_dir):
                    next_month_dir = os.path.join(next_year_dir, month_folder)
                    if os.path.isdir(next_month_dir):
                        # Percorrer os vídeos
                        for video_folder in os.listdir(next_month_dir):
                            next_video_dir = os.path.join(next_month_dir, video_folder)
                            if os.path.isdir(next_video_dir):   
                                # Tentar abrir o arquivo csv
                                try:
                                    df_estatisticas = pd.read_csv(f'{next_video_dir}/estatisticas_video.csv')
                                    lista_indexes.append(index)
                                    index += 1
                                    lista_media.append(df_estatisticas['media'].iloc[0])
                                    lista_desvio_padrao.append(df_estatisticas['desvio_padrao'].iloc[0])
                                    lista_mediana.append(df_estatisticas['mediana'].iloc[0])
                                except Exception as e:
                                    print(f'Inválido: {next_video_dir}')

    # Plotar a média
    plt.plot(lista_indexes, lista_media, label='Média de Toxicidade', color='green')
    # Plotar o desvio padrão
    plt.plot(lista_indexes, lista_desvio_padrao, label='Desvio Padrão de Toxicidade', color='purple')
    # Plotar a mediana
    plt.plot(lista_indexes, lista_mediana, label='Mediana de Toxicidade', color='orange')
    # Definir os ticks do eixo x
    #plt.xticks(lista_indexes)
    # Definir os limites do eixo y
    plt.ylim(0.0, 1.0)
    # Adicionar rótulos e título
    plt.xlabel('Vídeos')
    plt.ylabel('Nível de Toxicidade')
    plt.title(f'Análise de Toxicidade do {youtuber} (Média, Desvio Padrão e Mediana)')
    # Adicionar grade
    plt.grid(True)
    # Adicionar legenda
    plt.legend()
    # Salvar o gráfico consolidado
    plt.savefig(f'{base_dir}/grafico_estatisticas.png')
    plt.close()

'''
    Função para percorrer as pastas de cada vídeo de um youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param function - Função a ser executada na pasta de cada vídeo de um youtuber
'''
def percorrer_video(youtubers_list: list[str], function) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            console.print(f">>>" + base_dir)
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):  
                                    graficos_dir = os.path.join(next_video_dir, 'graficos')
                                    if not os.path.isdir(graficos_dir):
                                        os.mkdir(graficos_dir)
                                    try:
                                        df_tiras_video = pd.read_csv(f'{next_video_dir}/tiras_video.csv')
                                        if len(df_tiras_video['toxicidade']) > 2:
                                            function(next_video_dir)
                                    except Exception as e:
                                        console.print(f'Inválido: {e}')

'''
    Função para gerar os gráficos de distribuição temporal da toxicidade de cada vídeo 
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_linha(video_dir: str) -> None:
    try:
        # Tentar abrir o arquivo csv
        df_tiras = pd.read_csv(f'{video_dir}/tiras_video.csv')
        lista_toxicidade = df_tiras['toxicidade']
        indexes = list(range(1, len(lista_toxicidade) + 1))

        # Testar se o vídeo possui apenas uma tira
        if len(lista_toxicidade) == 1:
            plt.bar(['1'], lista_toxicidade, width=0.5)
        else:
            plt.plot(indexes, lista_toxicidade, label='Toxicidade no Vídeo')
            plt.xticks(indexes)
            plt.ylim(0.0, 1.0)
            plt.legend()

        # Configurações em comum
        plt.xlabel('Tiras')
        plt.ylabel('Toxicidade')
        plt.title('Análise da Oscilação de Toxicidade')
        plt.grid(True)
        plt.savefig(f'{video_dir}/graficos/grafico_toxicidade.png')
        plt.close()       
        
        # Calcular métricas do vídeo
        video_id = calcular_metricas_video(video_dir, lista_toxicidade)

        # Gerar gráfico HeatMap do vídeo
        if video_id != None and len(lista_toxicidade) > 1:
            gerar_grafico_heatmap(video_id, video_dir, indexes, lista_toxicidade)
    except Exception as e:
        #console.print(f'Inválido: {video_dir}')
        pass

'''
    Função para calcular a média, o desvio padrão e a mediana de toxicidade de um vídeo
    @param video_dir - Caminho para a pasta do vídeo
    @param lista_toxicidade - Lista da toxicidade de cada tira do vídeo
'''
def calcular_metricas_video(video_dir: str, lista_toxicidade: pd.core.series.Series) -> str:
    try:
        # Calcular métricas
        df_videos_info = pd.read_csv(f'{video_dir}/videos_info.csv')
        video_id = df_videos_info['video_id'].iloc[0]
        media = np.mean(lista_toxicidade)
        desvio_padrao = np.std(lista_toxicidade)
        mediana = np.median(lista_toxicidade)
        # Salvar métricas como arquivo csv
        estatisticas = {
            'id_video': [video_id],
            'media': [media],
            'desvio_padrao': [desvio_padrao],
            'mediana': [mediana]
        }
        df_estatisticas = pd.DataFrame(estatisticas)
        df_estatisticas.to_csv(f'{video_dir}/estatisticas_video.csv', index=False)
        return video_id
    except Exception as e:
        console.print(f'Inválido: {e}')
        return None

'''
    Função para gerar o gráfico de HeatMap da toxicidade de um vídeo
    @param video_id - Identificador do vídeo analisado
    @param video_dir - Pasta do youtuber para salvar o gráfico
    @param indexes - Lista de indexes de 1...N, sendo N a quantidade total de tiras
    @param lista_toxicidade - Lista da toxicidade de cada tira do vídeo
'''
def gerar_grafico_heatmap(video_id: str, video_dir: str, indexes: list, lista_toxicidade: pd.core.series.Series) -> None:
    # Preparar os dados para o formato de matriz 2D que o imshow precisa
    lista_heatmap = np.array(lista_toxicidade).reshape(1, -1)

    # Criar figura com tamanho padrão
    fig, ax = plt.subplots(figsize=(12, 2.5))

    # Definir um limite máximo para a barra de cores (ex: 1.0) para que a escala seja consistente entre gráficos
    limite_max_toxicidade = 1.0
    im = ax.imshow(lista_heatmap, cmap='Reds', aspect='auto', vmin=0, vmax=limite_max_toxicidade)

    # Adicionar a barra de cores
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label('Nível de Toxicidade')

    # Personalizar títulos e eixos
    ax.set_title(f'Heatmap de Toxicidade: {video_id}', fontsize=12)
    ax.set_xlabel('Segmento de Tempo (minuto)', fontsize=10)
    
    # Configurar os marcadores do eixo X para mostrar os números dos segmentos
    ax.set_xticks(np.arange(len(indexes)))
    ax.set_xticklabels(indexes, fontsize=8)

    # Remover o eixo Y que não tem significado nesse contexto
    ax.get_yaxis().set_visible(False)

    # Salvar gráfico
    plt.savefig(f'{video_dir}/graficos/grafico_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

'''
    Função para gerar os gráficos Facet Grid de um youtuber em linhas e em heatmap
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_facet_grid(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            console.print(f">>> " + base_dir)
            # Criar o Dict com os dados dos vídeos (video_id, lista de toxicidade das tiras)
            dados_videos = {}
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):  
                                    try:
                                        # Encontrar o video_id
                                        df_videos_info = pd.read_csv(f'{next_video_dir}/videos_info.csv')
                                        video_id = df_videos_info['video_id'].iloc[0]

                                        # Encontrar a toxicidade
                                        df_tiras = pd.read_csv(f'{next_video_dir}/tiras_video.csv')
                                        lista_toxicidade = df_tiras['toxicidade']

                                        # Testar se o vídeo não é um Short
                                        if len(lista_toxicidade) > 2:
                                            # Adicionar ao Dict
                                            dados_videos.setdefault(video_id, lista_toxicidade.tolist())
                                    except Exception as e:
                                        console.print(f'Inválido: {e}')

            # Gerar o gráfico Facet Grid com linhas
            gerar_grafico_facet_grid_linhas(base_dir, dados_videos)

            # Gerar o gráfico Facet Grid com heatmap
            gerar_grafico_facet_grid_heatmap(base_dir, dados_videos)

'''
    Função para gerar o gráfico Facet Grid de um youtuber em linhas
    @param base_dir - Pasta para salvar o gráfico
    @param dados_videos - Dicionário com os pares (video_id, lista de toxicidade das tiras do vídeo)
'''                                   
def gerar_grafico_facet_grid_linhas(base_dir: str, dados_videos: dict) -> None:
    # Checar se o dicionário ou lista está vazio
    if not dados_videos: 
        console.print("Nenhum dado de vídeo encontrado para gerar o gráfico. Pulando esta etapa.")
        return 
    
    # Configurar a grade do gráfico
    num_videos = len(dados_videos)
    cols = 3
    rows = math.ceil(num_videos / cols)

    # Criar a figura e a grade de subplots (`sharey=True` é um atalho para garantir que todos os gráficos na mesma linha compartilhem o eixo Y)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharey=True)

    # Achatar o array de eixos para facilitar a iteração com um único loop
    axes = axes.flatten()

    # Percorrer cada par chave-valor
    for i, (video_id, toxicidades) in enumerate(dados_videos.items()):
        # Selecionar o subplot atual
        ax = axes[i]
        
        # Criar lista de indexes de cada vídeo
        lista_indexes = range(1, len(toxicidades) + 1)
        # Plotar os dados de toxicidade no subplot
        ax.plot(lista_indexes, toxicidades)
        
        # Definir um título para cada subplot
        ax.set_title(video_id, fontsize=10)
        
        # Definir o mesmo limite do eixo Y para TODOS os gráficos. Isso garante uma comparação visual justa.
        ax.set_ylim(0, 1.0)
        
        # Adicionar uma grade de fundo para facilitar a leitura dos valores
        ax.grid(True, linestyle='--', alpha=0.6)

    # Esconder os eixos dos subplots que não foram usados (se houver)
    for i in range(num_videos, len(axes)):
        axes[i].set_visible(False)

    # Adicionar um título geral para a figura inteira
    fig.suptitle("Grid de Análise de Toxicidade por Vídeo", fontsize=16, y=0.98)

    # Adicionar rótulos comuns para os eixos X e Y
    fig.supxlabel("Duração do Vídeo (em tiras de 1 minuto)", fontsize=12)
    fig.supylabel("Nível de Toxicidade", fontsize=12)

    # Ajustar o layout para evitar sobreposição (`rect` deixa espaço para o suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salvar gráfico
    plt.savefig(f'{base_dir}/linhas_facet_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

'''
    Função para gerar o gráfico Facet Grid de um youtuber em heatmap
    @param base_dir - Pasta para salvar o gráfico
    @param dados_videos - Dicionário com os pares (video_id, lista de toxicidade das tiras do vídeo)
'''  
def gerar_grafico_facet_grid_heatmap(base_dir: str, dados_videos: dict) -> None:
    # Checar se o dicionário ou lista está vazio
    if not dados_videos: 
        console.print("Nenhum dado de vídeo encontrado para gerar o gráfico. Pulando esta etapa.")
        return 
    
    # Configurar a grade do gráfico
    num_videos = len(dados_videos)
    cols = 3
    rows = math.ceil(num_videos / cols)

    # Criar a figura e a grade de subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.5 * rows), constrained_layout=True)

    # Achatar o array de eixos para facilitar a iteração com um único loop
    axes = axes.flatten()

    # Variável para guardar o objeto de imagem, necessário para a barra de cor
    im = None 

    # Percorrer cada par chave-valor
    for i, (video_id, lista_toxicidade) in enumerate(dados_videos.items()):
        # Selecionar o subplot atual
        ax = axes[i]

        #Remodelar os dados para o formato de uma matriz 2D que o imshow precisa
        dados_heatmap = np.array(lista_toxicidade).reshape(1, -1)

        # Plotar os dados de toxicidade no subplot
        im = ax.imshow(dados_heatmap, cmap='Reds', aspect='auto', vmin=0, vmax=1.0)
        
        # Definir um título para cada subplot
        ax.set_title(video_id, fontsize=10)

        # Definir quantidade de tiras
        num_tiras = len(lista_toxicidade)
        
        # Se o vídeo for curto, mostrar todos os rótulos
        if num_tiras <= 20: 
            posicoes_ticks = np.arange(num_tiras)
            rotulos_ticks = list(range(1, num_tiras + 1))
        # Se for longo, mostrar no máximo 10 rótulos bem espaçados
        else:
            num_rotulos = 10
            posicoes_ticks = np.linspace(0, num_tiras - 1, num=num_rotulos).astype(int)
            rotulos_ticks = posicoes_ticks + 1

        # Definir posição dos marcadores
        ax.set_xticks(posicoes_ticks)
        ax.set_xticklabels(rotulos_ticks, fontsize=8)

        # O eixo Y não é informativo para este tipo de heatmap, então é removido
        ax.get_yaxis().set_visible(False)
        
        # Ajustar os ticks do eixo X para não poluir
        ax.tick_params(axis='x', labelsize=8)

    # Esconder os eixos dos subplots que não foram usados (se houver)
    for i in range(num_videos, len(axes)):
        axes[i].set_visible(False)

    # Adiciona um título geral para a figura
    fig.suptitle("Grid de Heatmaps de Toxicidade por Vídeo", fontsize=16, y=1.02)
    fig.supxlabel("Duração do Vídeo (em tiras de 1 minuto)", fontsize=12)

    #Adicionar UMA ÚNICA barra de cor para a figura inteira
    # `ax=axes.tolist()`: informa à barra de cor para "roubar" espaço de todos os subplots
    # `shrink` e `pad` são ajustes finos de posicionamento e tamanho
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Nível de Toxicidade", fontsize=12)

    # Salvar gráfico
    plt.savefig(f'{base_dir}/heatmap_facet_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

'''
    Função para gerar o gráfico de toxicidade dos percentis individuais de cada vídeo
    @param video_dir - Pasta do vídeo para salvar o gráfico
'''
def salvar_percentil_individual(video_dir: str) -> None:
    try:
        # Encontrar a toxicidade
        df_tiras = pd.read_csv(f'{video_dir}/tiras_video.csv')
        lista_toxicidade = df_tiras['toxicidade']

        # Encontrar o número de tiras do vídeo
        num_tiras = len(lista_toxicidade)

        # Definir os valores do eixo Y originais
        coordenadas_originais = lista_toxicidade

        # Encontrar as posições em percentis das tiras originais (np.linspace cria N pontos igualmente espaçados de 0 a 100)
        abscissas_originais = np.linspace(0, 100, num=num_tiras)

        # Encontrar as posições dos percetis desejados
        abscissas_percentis = np.arange(101) # Cria um array [0, 1, 2, ..., 100]

        # Interpolar para encontrar valores intermediários (np.interp(onde_queremos_saber, posicoes_originais, valores_originais))
        coordenadas_percentis = np.interp(abscissas_percentis, abscissas_originais, coordenadas_originais)

        # Estruturar os dados em um formato tabular com colunas claras
        dados_para_salvar = {
            'percentil': abscissas_percentis,
            'toxicidade_normalizada': coordenadas_percentis
        }
        df_percentis = pd.DataFrame(dados_para_salvar)

        # Definir o caminho do arquivo de saída
        caminho_csv = f'{video_dir}/dados_percentis_normalizados.csv'

        # Salvar o DataFrame em um arquivo CSV
        df_percentis.to_csv(caminho_csv, index=False, float_format='%.8f')
    
        # Criar o gráfico de visualização
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plota a curva interpolada, que é suave e tem 101 pontos
        ax.plot(abscissas_percentis, coordenadas_percentis, label='Curva de Percentis (Interpolada)', color='blue', linewidth=2.5)

        # Plota os pontos de dados originais para vermos como a interpolação se comportou
        ax.plot(abscissas_originais, coordenadas_originais, 'o--', label=f'Dados Originais ({num_tiras} tiras)',       color='red', alpha=0.7)

        # Configurações do gráfico
        ax.set_title('Análise de Toxicidade por Percentil de Duração do Vídeo', fontsize=16)
        ax.set_xlabel('Percentil de Duração do Vídeo (%)', fontsize=12)
        ax.set_ylabel('Nível de Toxicidade', fontsize=12)
        ax.set_xlim(0, 100) # Garante que o eixo X vá de 0 a 100
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.savefig(f'{video_dir}/graficos/grafico_percentis_individuais.png')
        plt.close()                               
    except Exception as e:
        console.print(f'Inválido: {e}')

'''
    Função para gerar o gráfico de toxicidade dos percentis agrupados de cada vídeo
    @param video_dir - Pasta do vídeo para salvar o gráfico
'''
def salvar_percentil_agrupado(video_dir: str) -> None:
    try:
        # Encontrar o identificador do vídeo
        df_videos_info = pd.read_csv(f'{video_dir}/videos_info.csv')
        video_id = df_videos_info['video_id'].iloc[0]

        # Encontrar a toxicidade
        df_tiras = pd.read_csv(f'{video_dir}/tiras_video.csv')
        lista_toxicidade = df_tiras['toxicidade']

        # Encontrar o número de tiras do vídeo
        num_tiras = len(lista_toxicidade)
        
        # Testar se há menos tiras que o número de grupos
        num_grupos = 5
        if num_tiras < num_grupos:
            num_grupos = num_tiras // 3 + 1
            if num_grupos == 0: num_grupos = 1

        # Criar novo DataFrame para fazer o agrupamento
        df_tiras_grupos = pd.DataFrame({'toxicidade': lista_toxicidade})
        
        # Adicionar uma coluna com a posição percentual de cada tira. 
        # (df_tiras_grupos.index / (num_tiras - 1)) garante que a escala vá de 0 a 1. Em seguida, multiplica-a por 100.
        if num_tiras > 1:
            df_tiras_grupos['percentil'] = (df_tiras_grupos.index / (num_tiras - 1)) * 100
        else: # Caso de vídeo com apenas 1 tira
            df_tiras_grupos['percentil'] = 50.0

        # Definir os liites superiores dos grupos. Ex: [0, 5, 10, ..., 100]
        step = 100 / num_grupos
        limites_superiores = np.arange(0, 101, step)
        
        # Criar os rótulos para cada grupo. Ex: '0-5%', '5-10%'
        grupos_labels = [f'{int(i)}-{int(i + step)}%' for i in limites_superiores[:-1]]
        
        # Colocar cada tira em seu respectivo grupo
        df_tiras_grupos['grupo_temporal'] = pd.cut(df_tiras_grupos['percentil'], 
                                                bins=limites_superiores, 
                                                labels=grupos_labels, 
                                                include_lowest=True)

        # Usar groupby() para agrupar por grupo e .agg() para calcular a média, desvio padrão e contagem.
        # .dropna() remove grupos que porventura ficaram vazios.
        metricas_por_grupos = df_tiras_grupos.groupby('grupo_temporal', observed=True)['toxicidade'].agg(['mean', 'std', 'count']).dropna().reset_index()

        # Adicionar o ID do vídeo ao DataFrame de métricas
        metricas_por_grupos['video_id'] = video_id

        # Definir o caminho do arquivo de saída
        caminho_csv = f'{video_dir}/estatisticas_percentis_agrupados.csv'

        # Salvar o DataFrame em um arquivo CSV
        metricas_por_grupos.to_csv(caminho_csv, index=False, float_format='%.8f')

        # Criar o gráfico de barras para a média de cada grupo
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(x=metricas_por_grupos['grupo_temporal'], 
            height=metricas_por_grupos['mean'], 
            color='skyblue', 
            label='Toxicidade Média por Bin')

        # Adicionar barras de erro para mostrar a variabilidade (desvio padrão) dentro de cada grupo
        ax.errorbar(x=metricas_por_grupos['grupo_temporal'], 
                    y=metricas_por_grupos['mean'], 
                    yerr=metricas_por_grupos['std'], 
                    fmt='none', # Não mostra marcador, apenas a barra de erro
                    ecolor='darkred', 
                    capsize=5, # Tamanho da "tampa" da barra de erro
                    label='Desvio Padrão')

        # Configurações do Gráfico
        ax.set_title(f'Análise de Toxicidade por Grupos Temporais\nID do Vídeo: {video_id}', fontsize=16)
        ax.set_ylabel('Nível de Toxicidade', fontsize=12)
        ax.set_xlabel('Percentil de Duração do Vídeo (Agrupado)', fontsize=12)
        ax.set_ylim(0, 1) # Mantém a escala de 0 a 1
        plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para não sobrepor
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()        
        plt.tight_layout()
        plt.savefig(f'{video_dir}/graficos/grafico_percentis_agrupados.png')
        plt.close()
    except Exception as e:
        console.print(f'Inválido: {e}')

'''
    Função para gerar o gráfico geral de toxicidade dos percentis individuais de cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_geral_percentil_individual(youtubers_list: list[str]) -> None:
    # Definir estrutura de dados
    matriz_toxicidade = []
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            console.print(f">>>" + base_dir)
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):  
                                    try:
                                        df_dados_percentis_normalizados = pd.read_csv(f'{next_video_dir}/dados_percentis_normalizados.csv')
                                        lista_toxicidade = df_dados_percentis_normalizados['toxicidade_normalizada'].tolist()
                                        matriz_toxicidade.append(lista_toxicidade)
                                    except Exception as e:
                                        console.print(f'Inválido: {e}')

    # Testar se algum valor foi encontrado
    if not matriz_toxicidade:
        print("Nenhum dado de percentis individuais foi encontrado. Encerrando a função.")
        return

    # Calcular a média de cada percentil (coluna)
    media_por_percentil = np.mean(matriz_toxicidade, axis=0)

    # Calcular o desvio padrão de cada percentil (coluna)
    desvio_padrao_por_percentil = np.std(matriz_toxicidade, axis=0)

    # Criar o gráfico para visualização
    fig, ax = plt.subplots(figsize=(14, 8))

    # Definir o eixo X, que são os percentis de 0 a 100
    percentis = np.arange(101)

    # Plotar a linha da média
    ax.plot(percentis, media_por_percentil, label='Toxicidade Média Geral', color='blue', linewidth=2.5)

    # Plotar a área sombreada do desvio padrão (a função fill_between é boa para mostrar variabilidade)
    ax.fill_between(percentis, 
                    media_por_percentil - desvio_padrao_por_percentil, 
                    media_por_percentil + desvio_padrao_por_percentil, 
                    color='blue', 
                    alpha=0.2, # Deixa a área semi-transparente
                    label='Desvio Padrão')

    # Configurações do Gráfico
    ax.set_title('Tendência Temporal da Toxicidade Média (Todos os Vídeos)', fontsize=18)
    ax.set_xlabel('Percentil de Duração do Vídeo (%)', fontsize=14)
    ax.set_ylabel('Nível de Toxicidade', fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('files/grafico_geral_percentil_individual.png')
    plt.close()

'''
    Função para gerar o gráfico geral de toxicidade dos percentis agrupados de cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_geral_percentil_agrupado(youtubers_list: list[str]) -> None:
    # Definir estrutura de dados
    lista_dataframes = []
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            console.print(f">>>" + base_dir)
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):  
                                    try:
                                        df_estatisticas_percentis_agrupados = pd.read_csv(f'{next_video_dir}/estatisticas_percentis_agrupados.csv')
                                        lista_dataframes.append(df_estatisticas_percentis_agrupados)
                                    except Exception as e:
                                        console.print(f'Inválido: {e}')

    # Testar se algum valor foi encontrado
    if not lista_dataframes:
        print("Nenhum dado de estatísticas agrupadas foi encontrado. Encerrando a função.")
        return
    
    # Concatenar todos os DataFrames de cada vídeo em um único DataFrame mestre
    df_geral = pd.concat(lista_dataframes, ignore_index=True)

    # Para a média ponderada, calcula-se primeiro a soma total da toxicidade (mean * count) por grupo
    df_geral['soma_toxicidade'] = df_geral['mean'] * df_geral['count']
    
    # Agrupar por grupo temporal e somar a 'soma_toxicidade' e a 'count'
    somas_por_grupo = df_geral.groupby('grupo_temporal', observed=True).agg({
        'soma_toxicidade': 'sum', # toxicidade total do grupo
        'count': 'sum' # quantidade de tiras total do grupo
    })
    
    # A média ponderada é a soma total da toxicidade dividida pela contagem total de tiras
    media_geral_ponderada = somas_por_grupo['soma_toxicidade'] / somas_por_grupo['count']

    # Calcular o Desvio Padrão da média dos vídeos (isso mostra a variabilidade ENTRE os vídeos para cada grupo de tempo)
    desvio_padrao_geral_das_medias = df_geral.groupby('grupo_temporal', observed=True)['mean'].std()

    # Criar o gráfico para visualização
    fig, ax = plt.subplots(figsize=(14, 8))

    # O eixo X corresponde ao nome dos grupos
    grupos_temporais = media_geral_ponderada.index

    # Plotar a linha da média geral ponderada
    ax.plot(grupos_temporais, media_geral_ponderada, 
            label='Toxicidade Média Geral (Ponderada)', 
            color='darkred', 
            linewidth=2.5,
            marker='o') # Adiciona um marcador para cada grupo

    # b) Plotar a área sombreada do desvio padrão
    ax.fill_between(grupos_temporais, 
                    media_geral_ponderada - desvio_padrao_geral_das_medias, 
                    media_geral_ponderada + desvio_padrao_geral_das_medias, 
                    color='darkred', 
                    alpha=0.2, 
                    label='Desvio Padrão (entre vídeos)')

    # Configurações do Gráfico
    ax.set_title('Tendência Temporal da Toxicidade Média (Dados Agrupados)', fontsize=18)
    ax.set_xlabel('Percentil de Duração do Vídeo (Agrupado)', fontsize=14)
    ax.set_ylabel('Nível de Toxicidade', fontsize=14)
    ax.set_ylim(0, max(media_geral_ponderada.max() * 1.2, 0.5)) # Ajuste dinâmico do eixo Y
    ax.grid(True, linestyle='--', alpha=0.6)
    #plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('files/grafico_geral_percentil_agrupado.png', dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    #lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']

    lista_youtubers =  ['AuthenticGames', 'Cadres']

    #percorrer_video(lista_youtubers, gerar_grafico_linha)
    #estatisticas_youtuber('Kass e KR')

    # Gráficos do Lucas
    #percorrer_video(lista_youtubers, gerar_grafico_linha)
    #gerar_grafico_facet_grid(lista_youtubers)
    #percorrer_video(lista_youtubers, salvar_percentil_individual)
    #percorrer_video(lista_youtubers, salvar_percentil_agrupado)
    #gerar_grafico_geral_percentil_individual(lista_youtubers)
    gerar_grafico_geral_percentil_agrupado(lista_youtubers)

    # Gráficos do Augusto
    #salvar_tiras_total_youtubers(lista_youtubers)
    #gerar_graficos_tiras_media_dp(lista_youtubers)