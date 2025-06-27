import pandas as pd
import re
import os

'''
    Função para converter a duração do vídeo no formato entregue pela API do Youtube para segundos
    @param str_duration - Duração no formato antigo
    @return duration - Duração em segundos
'''
def convertDuration(str_duration: str) -> int:
    # Expressão regular para capturar minutos (M) e segundos (S)
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str_duration)
    
    # Testar se a string não atende ao padrão
    if not match:
        return -1
    
    horas = int(match.group(1)) if match.group(1) else 0
    minutos = int(match.group(2)) if match.group(2) else 0
    segundos = int(match.group(3)) if match.group(3) else 0

    return horas * 3600 + minutos * 60 + segundos

'''
    Função para atuar na pasta individual do vídeo
    @param dir_path - Caminho para a pasta do vídeo
'''
def videoFolderFuncion(dir_path: str, dados: list[list[int, int]], columns: list[str]):
    # Criar caminho para o arquivo 'videos_info.csv' dentro da pasta do vídeo
    csv_videos_path = os.path.join(dir_path, 'videos_info.csv')

    # Testar se o arquivo existe
    if not os.path.exists(csv_videos_path):
        print(f"Erro! [{csv_videos_path}] não existe! (5)")
        return
    
    # Definir dados
    df_video_info = pd.read_csv(csv_videos_path)

    # Remover as linhas que possuem valores inválidos
    df_video = df_video_info.dropna(subset=columns)

    # Separar as colunas de interesse
    columns_video = df_video[columns]

    # Separa a linha do vídeo
    linha = columns_video.iloc[0].tolist()

    # Tratar os dados da linha do vídeo
    linha_tratada = [convertDuration(linha[0]), int(linha[1]), int(linha[2]), int(linha[3]), df_video['video_id'].iloc[0]]

    # Adicionar a linha à lista principal
    dados.append(linha_tratada)    

'''
    Função para percorrer toda a estrutura de pastas dos arquivos coletados
    @param func - Função para ser aplicada dentro da pasta do vídeo
'''
def findVideoFolder(func, columns: list[str]):
    # Definir dados
    base_dir = "files"
    dados = []

    # Percorrer youtubers
    for ytb_folder in os.listdir(base_dir):
        # Criar caminho 'files/ytb_folder'
        next_ytb_dir = os.path.join(base_dir, ytb_folder)

        # Testar se a pasta existe
        if not os.path.isdir(next_ytb_dir):
            # print(f"Erro! [{next_ytb_dir}] não existe! (1)")
            continue

        # Percorrer as pastas dos anos (dentro de 'files/ytb_folder')
        for year_folder in os.listdir(next_ytb_dir):
            # Criar caminho 'files/ytb_folder/year_folder'
            next_year_dir = os.path.join(next_ytb_dir, year_folder)

            # Testar se a pasta existe
            if not os.path.isdir(next_year_dir):
                # print(f"Erro! [{next_year_dir}] não existe! (2)")
                continue

            # Percorrer as pastas dos meses (dentro de 'files/ytb_folder/year_folder')
            for month_folder in os.listdir(next_year_dir):
                # Criar caminho 'files/ytb_folder/year_folder/month_folder'
                next_month_dir = os.path.join(next_year_dir, month_folder)

                # Testar se a pasta existe
                if not os.path.isdir(next_month_dir):
                    # print(f"Erro! [{next_month_dir}] não existe! (3)")
                    continue

                # Percorrer as pastas dos vídeos (dentro de 'files/ytb_folder/year_folder/month_folder/video_folder')
                for video_folder in os.listdir(next_month_dir):
                    # Criar caminho 'files/ytb_folder/year_folder/month_folder/video_folder'
                    next_video_dir = os.path.join(next_month_dir, video_folder)

                    # Testar se a pasta existe
                    if not os.path.isdir(next_video_dir):
                        # print(f"Erro! [{next_video_dir}] não existe! (4)")
                        continue

                    # Chamar função passada como parâmetro para atuar na pasta do vídeo
                    func(next_video_dir, dados, columns)

    # Converter a lista para DataFrame
    df = pd.DataFrame(dados, columns=columns)  

    df2 = df.drop_duplicates(subset=["video_id"])

    # Salvar a lista gerada em um arquivo .csv
    df2.to_csv("kmeans/kmeans_video.csv", index=False)

    # Mostrar mensagem de conclusão
    print(f"Análise concluída com sucesso! ({len(dados)} linhas)")

# Testing
if __name__ == "__main__":
    # Definir colunas de interesse
    columns = ['duration', 'comment_count', 'view_count', 'like_count', 'video_id']

    # Chamar a função para percorrer as pastas
    findVideoFolder( videoFolderFuncion, columns )
