import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        print(f">>>>>>>>" + base_dir)
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
                                    if index == 52:
                                        print(df_estatisticas['id_video'].iloc[0])
                                        print(df_estatisticas['desvio_padrao'].iloc[0])
                                except Exception as e:
                                    print(f'Inválido: {next_video_dir}')

    # Plotar a média
    print(lista_indexes[51])
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
            print(f">>>" + base_dir)
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
                                    function(next_video_dir)

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
        plt.savefig(f'{video_dir}/grafico_toxicidade.png')
        plt.close()       
        
        # Calcular métricas do vídeo
        video_id = calcular_metricas_video(video_dir, lista_toxicidade)

        # Gerar gráfico HeatMap do vídeo
        if video_id != None:
            gerar_grafico_heatmap(video_id, video_dir, indexes, lista_toxicidade)
    except Exception as e:
        #print(f'Inválido: {video_dir}')
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
        print(f'Inválido: {e}')
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
    plt.savefig(f'{video_dir}/grafico_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Válido: {video_dir}')

'''
    Função para gerar os gráficos Facet Grid de um youtuber em linhas e em heatmap
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_facet_grid(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            print(f">>> " + base_dir)
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

                                        # Adicionar ao Dict
                                        dados_videos.setdefault(video_id, lista_toxicidade.tolist())
                                    except Exception as e:
                                        print(f'Inválido: {e}')

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
        
        # Plotar os dados de toxicidade no subplot
        ax.plot(toxicidades)
        
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
    for i, (video_id, toxicidades) in enumerate(dados_videos.items()):
        # Selecionar o subplot atual
        ax = axes[i]

        #Remodelar os dados para o formato de uma matriz 2D que o imshow precisa
        dados_heatmap = np.array(toxicidades).reshape(1, -1)

        # Plotar os dados de toxicidade no subplot
        im = ax.imshow(dados_heatmap, cmap='Reds', aspect='auto', vmin=0, vmax=1.0)
        
        # Definir um título para cada subplot
        ax.set_title(video_id, fontsize=10)
        
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

#lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']
lista_youtubers =  ['Cadres']

#   percorrer_video(lista_youtubers, gerar_grafico_linha)
#estatisticas_youtuber('Kass e KR')
gerar_grafico_facet_grid(lista_youtubers)