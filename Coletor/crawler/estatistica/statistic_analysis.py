import math
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

def count_folders_os_walk(path):
    '''Counts the total number of folders within a given path, including subfolders.'''
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
        base_dir = f'files/{youtuber}'
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
                                console.print(f'Videos {month_folder}/{year_folder}: {atual}/{total_month_videos} ')
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                if os.path.isdir(next_video_dir):                              
                                    # Tentar abrir o arquivo csv
                                    try:
                                        df_tiras = pd.read_csv(f'{next_video_dir}/tiras_video.csv')
                                        youtuber_data = pd.concat([youtuber_data, df_tiras],ignore_index=True)
                                    except Exception as e:
                                        print(f'Inválido (salvar_tiras_total_youtubers): {next_video_dir}')
                                        pass
                                atual += 1
            if(not youtuber_data.empty):
                tiras_total_path = os.path.join(base_dir, 'tiras_total.csv')
                youtuber_data.to_csv(tiras_total_path, index=False)                     

def gerar_graficos_tiras_media_dp(youtubers_list: list[str]) -> None:
    all_data = pd.DataFrame()
    for youtuber in youtubers_list:
        try:
            base_dir = f'files/{youtuber}'
            tiras_total_path = os.path.join(base_dir, 'tiras_total.csv')
            youtuber_data = pd.read_csv(tiras_total_path)
            
            if(not youtuber_data.empty):
                console.print(f'Gerando dados do [cyan]{youtuber}')
        
                construir_grafico_toxicidade_media_dp(youtuber_data,youtuber,limite_index=40)

                all_data = pd.concat([all_data, youtuber_data],ignore_index=True)
        except Exception as e:
            console.print(f'{youtuber} não possui csv válido')
    if (not all_data.empty):
        console.print(f'Gerando gráfico geral')
        construir_grafico_toxicidade_media_dp(all_data,'',limite_index=40)

def construir_grafico_toxicidade_media_dp(youtuber_data,youtuber,limite_index):
    base_dir = 'files'
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
        base_dir = f'files/{youtuber}'
        plt.title(f'Análise da Oscilação de Toxicidade média do {youtuber}')
    else:
        plt.title(f'Análise de Oscilação de Toxicidade média')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{base_dir}/grafico_toxicidade_medio_video.png')
    plt.close() 

    console.print('Gráfico [green]salvo')

'''
    Função para percorrer as pastas de cada vídeo de um youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param function - Função a ser executada na pasta de cada vídeo de um youtuber
'''
def percorrer_video(youtubers_list: list[str], function_to_run) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar um objeto Path para o diretório base do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se o diretório do youtuber não existir, pular para o próximo
        if not base_path.is_dir():
            continue

        print(f'>>> Processando {youtuber}')

        # Criar a pasta 'graficos' no nível do youtuber, se não existir.
        (base_path / 'graficos').mkdir(parents=True, exist_ok=True)

        # .rglob('tiras_video.csv') busca recursivamente por este arquivo em todas as subpastas.
        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            # O diretório do vídeo é a pasta "pai" do arquivo CSV encontrado
            video_path = tiras_csv_path.parent
            
            try:
                # Criar a pasta 'graficos' dentro da pasta do vídeo, se não existir
                (video_path / 'graficos').mkdir(parents=True, exist_ok=True)
                
                # Ler o arquivo CSV que já foi encontrado
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Testar se não é um Short
                if len(df_tiras_video['toxicidade']) > 2:
                    function_to_run(str(video_path))
            
            except Exception as e:
                print(f'Inválido (percorrer_video): {e}')

# ---------------- Estatísticas e Gráficos de um vídeo - INÍCIO ----------------

'''
    Função para salvar as estatísticas e gerar os gráficos de distribuição temporal da toxicidade de cada vídeo 
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def toxicidade_video(video_dir: str) -> None:
    try:        
        # Encontrar o identificador do vídeo
        df_video_info = pd.read_csv(f'{video_dir}/videos_info.csv')
        video_id = df_video_info['video_id'].iloc[0]

        # Tentar abrir o arquivo csv
        df_tiras = pd.read_csv(f'{video_dir}/tiras_video.csv')
        lista_toxicidade = df_tiras['toxicidade']
        indexes = list(range(1, len(lista_toxicidade) + 1))

        if video_id != None and len(lista_toxicidade) > 1:
            # Calcular e salvar métricas do vídeo
            salvar_metricas_video(video_id, video_dir, lista_toxicidade)

            # Gerar gráfico de linhas do vídeo
            gerar_grafico_linha(video_id, video_dir, indexes, lista_toxicidade) 

            # Gerar gráfico HeatMap do vídeo
            gerar_grafico_heatmap(video_id, video_dir, indexes, lista_toxicidade)

            print(f'Estatísticas e gráficos salvos com sucesso em \'{video_dir}\'!')
    except Exception as e:
        console.print(f'Inválido (gerar_grafico_linha): {video_dir}')

'''
    Função para salvar em um csv a média, o desvio padrão, a mediana e o coeficiente de variação de toxicidade de um vídeo
    @param video_dir - Caminho para a pasta do vídeo
    @param lista_toxicidade - Lista da toxicidade de cada tira do vídeo
'''
def salvar_metricas_video(video_id: str, video_dir: str, lista_toxicidade: pd.core.series.Series) -> None:
    try:
        # Calcular métricas
        media = np.mean(lista_toxicidade)
        desvio_padrao = np.std(lista_toxicidade)
        mediana = np.median(lista_toxicidade)
        coeficiente_variacao = np.inf if media == 0 else desvio_padrao / media

        # Salvar métricas como arquivo csv
        estatisticas = {
            'id_video': [video_id],
            'media': [media],
            'desvio_padrao': [desvio_padrao],
            'mediana': [mediana],
            'coeficiente_variacao': [coeficiente_variacao]
        }
        df_estatisticas = pd.DataFrame(estatisticas)
        df_estatisticas.to_csv(f'{video_dir}/estatisticas_video.csv', index=False)
    except Exception as e:
        console.print(f'Inválido (salvar_metricas_video): {e}')

'''
    Função para gerar o gráfico de linha da toxicidade de um vídeo
    @param video_id - Identificador do vídeo analisado
    @param video_dir - Pasta do youtuber para salvar o gráfico
    @param indexes - Lista de indexes de 1...N, sendo N a quantidade total de tiras
    @param lista_toxicidade - Lista da toxicidade de cada tira do vídeo
'''
def gerar_grafico_linha(video_id: str, video_dir: str, indexes: list, lista_toxicidade: pd.core.series.Series) -> None:
   # Testar se o vídeo possui apenas uma tira
    if len(lista_toxicidade) == 1:
        plt.bar(['1'], lista_toxicidade, width=0.5)
    else:
        plt.plot(indexes, lista_toxicidade, label='Toxicidade no Vídeo')
        plt.xticks(indexes)
        plt.ylim(0.0, 1.0)
        plt.legend()

    # Configurações do gráfico
    plt.xlabel('Tiras')
    plt.ylabel('Toxicidade')
    plt.title(f'Análise da Oscilação de Toxicidade do Vídeo {video_id}')
    plt.grid(True)
    plt.savefig(f'{video_dir}/graficos/grafico_toxicidade.png', dpi=150)
    plt.close()  

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
    ax.set_title(f'Heatmap de Toxicidade do Vídeo {video_id}', fontsize=12)
    ax.set_xlabel('Segmento de Tempo (minuto)', fontsize=10)
    
    # Configurar os marcadores do eixo X para mostrar os números dos segmentos
    ax.set_xticks(np.arange(len(indexes)))
    ax.set_xticklabels(indexes, fontsize=8)

    # Remover o eixo Y que não tem significado nesse contexto
    ax.get_yaxis().set_visible(False)

    # Salvar gráfico
    plt.savefig(f'{video_dir}/graficos/grafico_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------- Estatísticas e Gráficos de um vídeo - FIM ----------------

# ---------------- Gráficos Facet Grid de um youtuber - INÍCIO ----------------

'''
    Função para gerar os gráficos Facet Grid de um youtuber em linhas e em heatmap
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_graficos_facet_grid(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar caminho base Path para a pasta do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se não existir a pasta, continuar para o próximo youtuber
        if not base_path.is_dir():
            continue

        print(f'>>> Processando youtuber: {youtuber}')
        
        # Criar o dicionário para armazenar os dados dos vídeos
        dados_videos = {}

        # .rglob() é usado para encontrar todos os arquivos 'videos_info.csv'
        for info_csv_path in base_path.rglob('videos_info.csv'):
            # A pasta do vídeo é a pasta "pai" do arquivo encontrado
            video_path = info_csv_path.parent
            tiras_csv_path = video_path / 'tiras_video.csv'

            try:
                # Garante que o outro arquivo necessário também exista
                if not tiras_csv_path.is_file():
                    continue

                # Encontrar o video_id do primeiro arquivo
                df_videos_info = pd.read_csv(info_csv_path)
                video_id = df_videos_info['video_id'].iloc[0]

                # Encontrar a toxicidade do segundo arquivo
                df_tiras = pd.read_csv(tiras_csv_path)
                lista_toxicidade = df_tiras['toxicidade']

                # Testar se o vídeo não é um Short
                if len(lista_toxicidade) > 2:
                    dados_videos[video_id] = lista_toxicidade.tolist()
            
            except Exception as e:
                print(f'Inválido (gerar_graficos_facet_grid): {e}')

        # Chamar as funções de plotagem
        if dados_videos:
            print(f"Dados de {len(dados_videos)} vídeos coletados. Gerando gráficos...")
            base_dir_str = str(base_path) # Converte Path de volta para string, se necessário

            # Gerar o gráfico Facet Grid com linhas
            gerar_grafico_facet_grid_linhas(base_dir_str, dados_videos, youtuber)

            # Gerar o gráfico Facet Grid com heatmap de percentis individuais 
            gerar_grafico_facet_grid_heatmap_individual(base_dir_str, dados_videos, youtuber)

            # Gerar o gráfico Facet Grid com heatmap de percentis agrupados
            gerar_grafico_facet_grid_heatmap_agrupado(base_dir_str, dados_videos, youtuber)
            
            print(f"Funções de plotagem para {youtuber} foram chamadas com sucesso.")
        else:
            print(f"Nenhum vídeo válido (>2 tiras) encontrado para {youtuber}.")

'''
    Função para gerar o gráfico Facet Grid de um youtuber em linhas
    @param base_dir - Pasta para salvar o gráfico
    @param dados_videos - Dicionário com os pares (video_id, lista de toxicidade das tiras do vídeo)
'''                                   
def gerar_grafico_facet_grid_linhas(base_dir: str, dados_videos: dict, youtuber: str) -> None:
    # Checar se o dicionário ou lista está vazio
    if not dados_videos: 
        console.print('Nenhum dado de vídeo encontrado para gerar o gráfico. Pulando esta etapa.')
        return 
    
    # Configurar a grade do gráfico
    num_videos = len(dados_videos)
    cols = 4
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
    fig.suptitle('Grid de Análise de Toxicidade por Vídeo', fontsize=16, y=0.98)

    # Adicionar rótulos comuns para os eixos X e Y
    fig.supxlabel(f'Duração do Vídeo (em tiras de 1 minuto) do {youtuber}', fontsize=12)
    fig.supylabel('Nível de Toxicidade', fontsize=12)

    # Ajustar o layout para evitar sobreposição (`rect` deixa espaço para o suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salvar gráfico
    plt.savefig(f'{base_dir}/graficos/linhas_facet_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

'''
    Função para gerar o gráfico Facet Grid de um youtuber em heatmap com percentis individuais
    @param base_dir - Pasta para salvar o gráfico
    @param youtuber - Nome do youtuber sendo analisado
'''  
def gerar_grafico_facet_grid_heatmap_individual(base_dir: str, dados_videos: dict, youtuber: str) -> None:
    if not dados_videos:
        print(f'Nenhum dado de percentis normalizados encontrado para {youtuber}.')
        return

    # Criar o gráfico para visualização
    print(f"Gerando gráfico com dados de {len(dados_videos)} vídeos...")
    num_videos = len(dados_videos)
    cols = 4
    rows = math.ceil(num_videos / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.5 * rows), constrained_layout=True)
    axes = axes.flatten()
    im = None

    for i, (video_id, coordenadas_percentis) in enumerate(dados_videos.items()):
        ax = axes[i]
        dados_heatmap = np.array(coordenadas_percentis).reshape(1, -1)
        im = ax.imshow(dados_heatmap, cmap='Reds', vmin=0, vmax=1.0, aspect='auto')

        ax.set_title(video_id, fontsize=9)
        ax.get_yaxis().set_visible(False)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)

    # Limpeza final no gráfico
    for i in range(num_videos, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Heatmap de Toxicidade Normalizada por Vídeo do {youtuber}', fontsize=16)
    fig.supxlabel('Progresso Percentual da Duração do Vídeo', fontsize=12)

    if im:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02)
        cbar.set_label('Nível de Toxicidade (Interpolado)')

    # Salvar o gráfico
    base_path = Path(base_dir)
    pasta_graficos = base_path / 'graficos'
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    caminho_salvar = pasta_graficos / 'heatmap_facet_grid_percentil_individual.png'
    plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico Facet Grid de percentis individuais salvo em: {caminho_salvar}")

'''
    Função para gerar o gráfico Facet Grid de um youtuber em heatmap com percentis agrupados
    @param base_dir - Pasta para salvar o gráfico
    @param dados_videos - Dicionário com os pares (video_id, lista de toxicidade das tiras do vídeo)
'''  
def gerar_grafico_facet_grid_heatmap_agrupado(base_dir: str, dados_videos: dict, youtuber: str, num_grupos: int = 20) -> None:
    # Checar se o dicionário ou lista está vazio
    if not dados_videos: 
        console.print('Nenhum dado de vídeo encontrado para gerar o gráfico. Pulando esta etapa.')
        return 
    
    # Definir dados de configuração da grade
    num_videos = len(dados_videos)
    cols = 4
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
        num_tiras = len(lista_toxicidade)

        # Se o vídeo for muito curto para binarizar, apenas mostrar a média geral dele
        if num_tiras < 5: 
            # Encontrar a média geral
            media_geral = np.mean(lista_toxicidade) if lista_toxicidade else 0

            # Criar um 'heatmap' de 1 pixel com a cor da média
            im = ax.imshow([[media_geral]], cmap='Reds', vmin=0, vmax=1.0, aspect='auto')
            ax.set_xticks([])
        else:
            # Criar novo DataFrame para fazer o agrupamento
            df = pd.DataFrame({'toxicidade': lista_toxicidade})
            df['percentil'] = (df.index / (num_tiras - 1)) * 100

            # Definir os liites superiores dos grupos. Ex: [0, 5, 10, ..., 100]
            step = 100 / num_grupos
            limites_superiores = np.arange(0, 101, step)
            
            # Criar os rótulos para cada grupo. Ex: '0-5%', '5-10%'
            grupos_labels = range(num_grupos) 

            df['grupo_temporal'] = pd.cut(df['percentil'], bins=limites_superiores, labels=grupos_labels, include_lowest=True, right=False)
            
            # Calcular a média de toxicidade para cada grupo
            media_por_grupo = df.groupby('grupo_temporal', observed=True)['toxicidade'].mean()
            
            # Reindexar para garantir que todos os bins de 0 a num_grupos-1 existam, preenchendo vazios com 0
            media_por_grupo = media_por_grupo.reindex(grupos_labels, fill_value=0)

            # Remodelar os dados (1 linha, num_grupos colunas) e plotamos
            dados_heatmap = np.array(media_por_grupo.values).reshape(1, -1)

            # Plotar dados no gráfico
            im = ax.imshow(dados_heatmap, cmap='Reds', vmin=0, vmax=1.0, aspect='auto')

            # Rmemover os ticks do eixo X para uma visualização mais limpa
            ax.set_xticks([])

        # Configurações do subplot individual
        ax.set_title(video_id, fontsize=9)
        ax.get_yaxis().set_visible(False)

    # Limpeza final
    for i in range(num_videos, len(axes)):
        axes[i].set_visible(False)

    # Adicionar o título e o subtítulo do grid
    fig.suptitle(f'Heatmap de Toxicidade por Vídeo (em {num_grupos} Grupos Temporais) do {youtuber}', fontsize=16, y=1.02)
    fig.supxlabel('Progresso Temporal do Vídeo (de 0% a 100%, agrupado em seções)', fontsize=12)

    # Adicionar a barra de cor compartilhada, a 'régua' de cores para todos os gráficos
    if im:
        cbar = fig.colorbar(im, ax=axes.tolist(), orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('Toxicidade Média no Grupo')

    plt.savefig(f'{base_dir}/graficos/heatmap_facet_grid_agrupado.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------- Gráficos Facet Grid de um youtuber - FIM ----------------

# ---------------- Estatísticas e Gráficos dos percentis de um vídeo - INÍCIO ----------------

'''
    Função para gerar o gráfico de toxicidade dos percentis individuais de cada vídeo
    @param video_dir - Pasta do vídeo para salvar o gráfico
'''
def salvar_percentil_individual(video_dir: str) -> None:
    try:
        # Encontrar identificador do vídeo
        df_video_info = pd.read_csv(f'{video_dir}/videos_info.csv')
        video_id = df_video_info['video_id'].iloc[0]

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
        ax.set_title(f'Análise de Toxicidade por Percentil de Duração do Vídeo {video_id}', fontsize=16)
        ax.set_xlabel('Percentil de Duração do Vídeo (%)', fontsize=12)
        ax.set_ylabel('Nível de Toxicidade', fontsize=12)
        ax.set_xlim(0, 100) # Garante que o eixo X vá de 0 a 100
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        video_path = Path(video_dir)
        caminho_salvar = video_path / 'graficos' / 'grafico_percentis_individuais.png'
        plt.savefig(caminho_salvar, dpi=150)
        plt.close(fig)     
        print(f"Gráfico de percentil individual salvo com sucesso em: {caminho_salvar}")                          
    except Exception as e:
        console.print(f'Inválido (salvar_percentil_individual): {e}')

'''
    Função para gerar o gráfico de toxicidade dos decis individuais de cada vídeo
    @param video_dir - Pasta do vídeo para salvar o gráfico
'''
def salvar_decil_individual(video_dir: str) -> None:
    try:
        # Encontrar identificador do vídeo
        df_video_info = pd.read_csv(f'{video_dir}/videos_info.csv')
        video_id = df_video_info['video_id'].iloc[0]

        # Encontrar a toxicidade
        df_tiras = pd.read_csv(f'{video_dir}/tiras_video.csv')
        lista_toxicidade = df_tiras['toxicidade']

        # Encontrar o número de tiras do vídeo
        num_tiras = len(lista_toxicidade)

        # Definir os valores do eixo Y originais
        coordenadas_originais = lista_toxicidade

        # Encontrar as posições em percentis das tiras originais (np.linspace cria N pontos igualmente espaçados de 0 a 10)
        abscissas_originais = np.linspace(0, 10, num=num_tiras)

        # Encontrar as posições dos percetis desejados
        abscissas_decis = np.arange(11) # Cria um array [0, 1, 2, ..., 11]

        # Interpolar para encontrar valores intermediários (np.interp(onde_queremos_saber, posicoes_originais, valores_originais))
        coordenadas_decis = np.interp(abscissas_decis, abscissas_originais, coordenadas_originais)

        # Estruturar os dados em um formato tabular com colunas claras
        dados_para_salvar = {
            'percentil': abscissas_decis,
            'toxicidade_normalizada': coordenadas_decis
        }
        df_decis = pd.DataFrame(dados_para_salvar)

        # Definir o caminho do arquivo de saída
        caminho_csv = f'{video_dir}/dados_decis_normalizados.csv'

        # Salvar o DataFrame em um arquivo CSV
        df_decis.to_csv(caminho_csv, index=False, float_format='%.8f')
    
        # Criar o gráfico de visualização
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plota a curva interpolada, que é suave e tem 11 pontos
        ax.plot(abscissas_decis, coordenadas_decis, label='Curva de Decis (Interpolada)', color='blue', linewidth=2.5)

        # Plota os pontos de dados originais para vermos como a interpolação se comportou
        ax.plot(abscissas_originais, coordenadas_originais, 'o--', label=f'Dados Originais ({num_tiras} tiras)', color='red', alpha=0.7)

        # Configurações do gráfico
        ax.set_title(f'Análise de Toxicidade por Decis de Duração do Vídeo {video_id}', fontsize=16)
        ax.set_xlabel('Decil de Duração do Vídeo (%)', fontsize=12)
        ax.set_ylabel('Nível de Toxicidade', fontsize=12)
        ax.set_xlim(0, 10) # Garante que o eixo X vá de 0 a 10
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        video_path = Path(video_dir)
        caminho_salvar = video_path / 'graficos' / 'grafico_decis_individuais.png'
        plt.savefig(caminho_salvar, dpi=150)
        plt.close(fig)     
        print(f"Gráfico de decil individual salvo com sucesso em: {caminho_salvar}")                          
    except Exception as e:
        console.print(f'Inválido (salvar_decil_individual): {e}')

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
            num_grupos = num_tiras // 2 + 1 # Ajuste para garantir ao menos 1 tira por grupo em média
            if num_grupos < 1: num_grupos = 1
            print(f"Info: Vídeo {video_id} é curto. Usando {num_grupos} grupos em vez de 20.")

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
            label='Toxicidade Média por Grupo')

        # Adicionar barras de erro para mostrar a variabilidade (desvio padrão) dentro de cada grupo
        ax.errorbar(x=metricas_por_grupos['grupo_temporal'], 
                    y=metricas_por_grupos['mean'], 
                    yerr=metricas_por_grupos['std'], 
                    fmt='none', # Não mostra marcador, apenas a barra de erro
                    ecolor='darkred', 
                    capsize=5, # Tamanho da 'tampa' da barra de erro
                    label='Desvio Padrão')

        # Configurações do Gráfico
        ax.set_title(f'Análise de Toxicidade por Grupos Temporais\nID do Vídeo: {video_id}', fontsize=16)
        ax.set_ylabel('Nível de Toxicidade', fontsize=12)
        ax.set_xlabel(f'Percentil de Duração do Vídeo {video_id} (Agrupado)', fontsize=12)
        ax.set_ylim(0, 1) # Mantém a escala de 0 a 1
        plt.xticks(rotation=60, ha='right') # Rotaciona os rótulos do eixo X para não sobrepor
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()        
        plt.tight_layout()

        video_path = Path(video_dir)
        caminho_salvar = video_path / 'graficos' / 'grafico_percentis_agrupados.png'
        plt.savefig(caminho_salvar, dpi=150)
        plt.close(fig)     
        print(f"Gráfico de percentil agrupado salvo com sucesso em: {caminho_salvar}")  
    except Exception as e:
        console.print(f'Inválido (salvar_percentil_agrupado): {e}')

'''
    Função para gerar o gráfico geral de toxicidade dos percentis individuais de cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_geral_percentil_individual(youtubers_list: list[str]) -> None:
    # Coletar os dados
    matriz_toxicidade = []

    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')
        
        if not base_path.is_dir():
            print(f"Diretório não encontrado para {youtuber}, pulando.")
            continue
            
        print(f'>>> Processando {base_path}')
        
        # .rglob() busca o arquivo 'dados_percentis_normalizados.csv' de forma recursiva
        for csv_path in base_path.rglob('dados_percentis_normalizados.csv'):
            try:
                df = pd.read_csv(csv_path)
                lista_toxicidade = df['toxicidade_normalizada'].tolist()
                
                # Verificação de segurança para garantir a integridade da matriz
                if len(lista_toxicidade) == 101:
                    matriz_toxicidade.append(lista_toxicidade)
                else:
                    print(f"Aviso: Arquivo em {csv_path.parent} não continha 101 pontos e foi ignorado.")
            
            except Exception as e:
                print(f'Inválido ao ler {csv_path}: {e}')

    # Calcular e plotar
    if not matriz_toxicidade:
        print('Nenhum dado de percentis individuais foi encontrado. Encerrando a função (gerar_grafico_geral_percentil_individual).')
        return
    
    print(f"\nDados de {len(matriz_toxicidade)} vídeos coletados. Calculando métricas e gerando gráfico...")
    
    # Calcular a média e o desvio padrão de cada percentil (coluna)
    media_por_percentil = np.mean(matriz_toxicidade, axis=0)
    desvio_padrao_por_percentil = np.std(matriz_toxicidade, axis=0)

    # Criar o gráfico para visualização
    fig, ax = plt.subplots(figsize=(14, 8))
    percentis = np.arange(101)

    # Plotar a linha da média
    ax.plot(percentis, media_por_percentil, label='Toxicidade Média Geral', color='blue', linewidth=2.5)

    # Plotar a área sombreada do desvio padrão
    ax.fill_between(percentis, 
                    media_por_percentil - desvio_padrao_por_percentil, 
                    media_por_percentil + desvio_padrao_por_percentil, 
                    color='blue', 
                    alpha=0.2,
                    label='Desvio Padrão')

    # Configurações do Gráfico
    ax.set_title('Tendência Temporal da Toxicidade Média de Todos os Vídeos', fontsize=18)
    ax.set_xlabel('Percentil de Duração do Vídeo (%)', fontsize=14)
    ax.set_ylabel('Nível de Toxicidade', fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    
    # Garantir que a pasta 'files' exista antes de salvar
    os.makedirs('files', exist_ok=True)
    caminho_salvar = 'files/grafico_geral_percentil_individual.png'
    plt.savefig(caminho_salvar, dpi=150)
    plt.close(fig)
    print(f"Gráfico geral de percentil individual salvo com sucesso em: {caminho_salvar}")

'''
    Função para gerar o gráfico geral de toxicidade dos percentis agrupados de cada vídeo
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_grafico_geral_percentil_agrupado(youtubers_list: list[str]) -> None:
    # Coletar os dados
    lista_dataframes = []
    print("Iniciando a leitura dos arquivos de estatísticas agrupadas...")
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        # Criar caminho base Path para a pasta do youtuber
        base_path = Path(f'files/{youtuber}')

        # Se não há a pasta do youtuber, continue para o próximo
        if not base_path.is_dir():
            continue
            
        print(f'>>> Processando {youtuber}')
        
        # .rglob() busca o arquivo 'estatisticas_percentis_agrupados.csv' de forma recursiva
        for csv_path in base_path.rglob('estatisticas_percentis_agrupados.csv'):
            try:
                df_estatisticas = pd.read_csv(csv_path)
                lista_dataframes.append(df_estatisticas)
            except Exception as e:
                print(f'Inválido (gerar_grafico_geral_percentil_agrupado): {e}')

    # Testar se algum valor foi encontrado
    if not lista_dataframes:
        print('Nenhum dado de estatísticas agrupadas foi encontrado. Encerrando a função.')
        return
    
    print(f"Dados de {len(lista_dataframes)} arquivos carregados. Agregando e gerando gráfico...")
    
    # Concatenar todos os DataFrames de cada vídeo em um único DataFrame mestre
    df_geral = pd.concat(lista_dataframes, ignore_index=True)

    # Calcular da média ponderada
    df_geral['soma_toxicidade'] = df_geral['mean'] * df_geral['count']

    # Calcular a soma total de cada grupo temporal
    somas_por_grupo = df_geral.groupby('grupo_temporal', observed=True).agg({
        'soma_toxicidade': 'sum',
        'count': 'sum'
    })

    # Calcular a média geral ponderada para cada um dos grupos
    media_geral_ponderada = somas_por_grupo['soma_toxicidade'] / somas_por_grupo['count']

    # Cálcular o Desvio Padrão da média dos vídeos
    desvio_padrao_geral_das_medias = df_geral.groupby('grupo_temporal', observed=True)['mean'].std()

    # Criar o gráfico para visualização
    fig, ax = plt.subplots(figsize=(15, 8))
    grupos_temporais = media_geral_ponderada.index

    # Plotar as barras com a altura da média
    ax.bar(x=grupos_temporais, 
           height=media_geral_ponderada, 
           color='skyblue', 
           alpha=0.8,
           label='Toxicidade Média Ponderada')

    # Adicionar as barras de erro para o desvio padrão
    ax.errorbar(x=grupos_temporais, 
                y=media_geral_ponderada, 
                yerr=desvio_padrao_geral_das_medias,
                fmt='none',
                ecolor='black', 
                capsize=5,
                label='Desvio Padrão (entre vídeos)')

    # Configurações do Gráfico
    ax.set_title('Tendência Temporal da Toxicidade Média de Todos os Vídeos (Dados Agrupados)', fontsize=18)
    ax.set_xlabel('Percentil de Duração do Vídeo (Agrupado)', fontsize=14)
    ax.set_ylabel('Nível de Toxicidade', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Garantir que a pasta 'files' exista antes de salvar
    os.makedirs('files', exist_ok=True)
    caminho_salvar = 'files/grafico_geral_percentil_agrupado.png'
    plt.savefig(caminho_salvar, dpi=150)
    plt.close(fig)
    print(f"Gráfico geral de barras salvo com sucesso em: {caminho_salvar}")

# ---------------- Estatísticas e Gráficos dos percentis de um vídeo - FIM ----------------

'''
    Função para organizar o coeficiente de variação dos vídeos de cada youtuber em um gráfico
    @param youtubers_list: Lisa de youtubers a serem analisados
'''
def organizar_coeficiente_variacao(youtubers_list: list[str]) -> None:
    # Percorrer os youtubers
    for youtuber in youtubers_list:
        lista_coeficiente_variacao = []
        base_dir = f'files/{youtuber}'
        base_path = Path(base_dir) # Criar um objeto Path para o diretório base

        # Se o diretório do youtuber não existir, pula para o próximo
        if not base_path.is_dir():
            continue

        print(f'>>> Coletando dados para {youtuber}')
        
        # .rglob('nome_do_arquivo') busca recursivamente em todas as subpastas
        for csv_path in base_path.rglob('estatisticas_video.csv'):
            try:
                df_estatisticas = pd.read_csv(csv_path)
                # Adicionar o valor à lista, garantindo que a coluna exista
                if 'coeficiente_variacao' in df_estatisticas.columns:
                    lista_coeficiente_variacao.append(df_estatisticas['coeficiente_variacao'].iloc[0])
            except Exception as e:
                # csv_path.parent é a pasta do vídeo
                print(f'Inválido ao processar o vídeo em {csv_path.parent}: {e}')

        # Testar se encontrou algum resultado
        if not lista_coeficiente_variacao:
            print(f"Nenhum dado de coeficiente de variação encontrado para {youtuber}.")
            continue # Pula para o próximo youtuber
        
        indexes = range(1, len(lista_coeficiente_variacao) + 1)

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(indexes, lista_coeficiente_variacao, label='Coeficiente de Variação', marker='o', linestyle='-')
        #ax.set_xticks(indexes)
        ax.set_ylim(0.0, max(lista_coeficiente_variacao) * 1.2)
        ax.legend()
        ax.set_xlabel('Vídeos (em ordem de processamento)')
        ax.set_ylabel('Coeficiente de Variação')
        ax.set_title(f'Análise do Coeficiente de Variação de Toxicidade dos Vídeos do {youtuber}')
        ax.grid(True, linestyle='--')
        
        # Garantir que a pasta de gráficos exista
        pasta_graficos = base_path / 'graficos'
        pasta_graficos.mkdir(parents=True, exist_ok=True)
        
        caminho_salvar = pasta_graficos / 'coeficiente_variacao.png'
        plt.savefig(caminho_salvar, dpi=150)
        plt.close(fig)
        print(f"Gráfico geral de coeficiente de variação salvo com sucesso em: {caminho_salvar}")
    

if __name__ == '__main__':
    #lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']

    #lista_youtubers =  ['AuthenticGames', 'Cadres']

    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']


    # Gráficos do Lucas
    #percorrer_video(lista_youtubers, toxicidade_video)
    percorrer_video(lista_youtubers, salvar_percentil_individual)
    #percorrer_video(lista_youtubers, salvar_decil_individual)
    percorrer_video(lista_youtubers, salvar_percentil_agrupado)
    gerar_graficos_facet_grid(lista_youtubers)
    #gerar_grafico_geral_percentil_individual(lista_youtubers)
    #gerar_grafico_geral_percentil_agrupado(lista_youtubers)
    #organizar_coeficiente_variacao(lista_youtubers)

    # Gráficos do Augusto
    #salvar_tiras_total_youtubers(lista_youtubers)
    #gerar_graficos_tiras_media_dp(lista_youtubers)