import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from rich.console import Console
from config import config



def comment_analysis(csv_path):
    # Pegar valores ordenados por data e limpar valores nulos
    comments_info = pd.read_csv(csv_path)

    # Adicionar o Interarrival Time dos comentarios
    comments_info['published_at'] = pd.to_datetime(comments_info['published_at'])
    comments_info = comments_info.sort_values('published_at')
    comments_info['inter-arrival-time'] = comments_info['published_at'].diff()
    comments_info['inter-arrival-time-seconds'] = comments_info['inter-arrival-time'].dt.total_seconds()
    comments_info.to_csv(csv_path, index=False)

    # Limpar os valores nulos da analise de sentimento
    # tam = len(comments_info)
    comments_info = comments_info.dropna(subset=['roberta-neg', 'roberta-pos'])
    # print("Ruido = "+str(tam-len(comments_info))+" comentários") 
    comments_sentimental_pos = comments_info['roberta-pos']
    threshold_pos = config['treshold'][0]
    
    comments_sentimental_neu = comments_info['roberta-neu']
    threshold_neu = config['treshold'][1]

    comments_sentimental_neg = comments_info['roberta-neg']
    threshold_neg = config['treshold'][2]
    

    comments_over_time = comments_info.resample('D', on='published_at').size()

    # Definir variavies a serem calculadas e inseridas no resultado da analise
    comments_total = len(comments_info.index)
    comments_mean_day = comments_over_time.mean()
    comments_avg_day = comments_over_time.std()
    comments_median_day = comments_over_time.median()
    comments_max_day = comments_over_time.max()
    filtered_rows = comments_sentimental_neg[comments_sentimental_neg > threshold_neg]
    neg_total_threshold = len(filtered_rows.index)
    neg_mean = comments_sentimental_neg.mean()
    neg_avg = comments_sentimental_neg.std()
    neg_max = comments_sentimental_neg.max() 
    neg_percentage =  (neg_total_threshold / len(comments_sentimental_neg.index)) * 100
    filtered_rows = comments_sentimental_pos[comments_sentimental_pos > threshold_pos]
    pos_total_threshold = len(filtered_rows.index)
    pos_mean = comments_sentimental_pos.mean()
    pos_avg = comments_sentimental_pos.std()
    pos_max = comments_sentimental_pos.max() 
    pos_percentage =  (pos_total_threshold / len(comments_sentimental_pos.index)) * 100
    filtered_rows = comments_sentimental_neu[comments_sentimental_neu > threshold_neu]
    neu_total_threshold = len(filtered_rows.index)
    neu_mean = comments_sentimental_neu.mean()
    neu_avg = comments_sentimental_neu.std()
    neu_max = comments_sentimental_neu.max() 
    neu_percentage =  (neu_total_threshold / len(comments_sentimental_neu.index)) * 100


    # Organize variables into a dictionary
    data = {
        'comments_total': [comments_total],
        'comments_mean_day': [comments_mean_day],
        'comments_avg_day': [comments_avg_day],
        'comments_median_day': [comments_median_day],
        'comments_max_day': [comments_max_day],
        'neg_total_threshold': [neg_total_threshold],
        'neg_mean': [neg_mean],
        'neg_avg': [neg_avg],
        'neg_max': [neg_max],
        'neg_percentage': [neg_percentage],
        'pos_total_threshold': [pos_total_threshold],
        'pos_mean': [pos_mean],
        'pos_avg': [pos_avg],
        'pos_max': [pos_max],
        'pos_percentage': [pos_percentage],
        'neu_total_threshold': [neu_total_threshold],
        'neu_mean': [neu_mean],
        'neu_avg': [neu_avg],
        'neu_max': [neu_max],
        'neu_percentage': [neu_percentage]
    }
    return data

def make_comment_over_time_graph(csv_path):
    comments_info = pd.read_csv(csv_path)
    comments_info['published_at'] = pd.to_datetime(comments_info['published_at'])
    comments_info = comments_info.sort_values('published_at')
    
    comments_over_time = comments_info.resample('D', on='published_at').size()
    plt.figure(figsize=(12, 6))
    plt.plot(comments_over_time.index, comments_over_time.values, marker='o', linestyle='-', color='b')

    plt.title('Number of Comments Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Comments', fontsize=14)

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    graph_path = folder_path + "/comment_over_time.png"
    plt.savefig(fname=graph_path)

def make_graph_neg_pos_comments(csv_path, folder_path, ytb_name):
    comments_info = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 6))

    # Plot neg_percentage
    plt.plot(comments_info.index, comments_info['neg_percentage'], marker='o', linestyle='-', color='red', label='Negative Percentage')

    # Plot pos_percentage
    plt.plot(comments_info.index, comments_info['pos_percentage'], marker='o', linestyle='-', color='green', label='Positive Percentage')

    # Plot pos_percentage
    plt.plot(comments_info.index, comments_info['neu_percentage'], marker='o', linestyle='-', color='gray', label='Neutral Percentage')
    
    # Add labels and title
    plt.title('Negative and Positive Percentages per Video of '+ytb_name, fontsize=16)
    plt.xlabel('Video', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)

    # Add a legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    graph_path = f"{folder_path}/{ytb_name}_comment_sentimental_analysis_graph.png"
    plt.savefig(fname=graph_path)

def atualizar_video_comentarios_coletados(nmCanal,total_videos, total_comentarios):
    youtuberListPath = "youtuberslist.csv"
    df = pd.read_csv(youtuberListPath)
    df.loc[df.nome == nmCanal, 'videosColetados'] = total_videos
    df.loc[df.nome == nmCanal, 'comentariosColetados'] = total_comentarios
    df.to_csv(youtuberListPath, index=False)

def main():
    base_dir = "files"
    console = Console()
    # comment_analysis("files/AuthenticGames/2020/Janeiro/BALDE DE MADEIRA !! - Minecraft Dinossauros #05/comments_info.csv")
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        videos_coletados = 0
        comentarios_coletados = 0
        ytb_data = pd.DataFrame()
        next_ytb_dir = os.path.join(base_dir, ytb_folder)
        if os.path.isdir(next_ytb_dir):
            # andar pelos anos
            for year_folder in os.listdir(next_ytb_dir):
                year_data = pd.DataFrame()
                next_year_dir = os.path.join(next_ytb_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # andar pelos meses
                    for month_folder in os.listdir(next_year_dir):
                        month_data = pd.DataFrame()
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # andar por cada pasta dentro do mes
                            for folder in os.listdir(next_month_dir):
                                folder_path = os.path.join(next_month_dir, folder)
                                if os.path.isdir(folder_path):
                                    # Path to the comments_info.csv file
                                    csv_path = os.path.join(folder_path, 'comments_info.csv')
                                    # Check if the file exists
                                    if os.path.exists(csv_path):
                                        result_df = pd.DataFrame(comment_analysis(csv_path))
                                        video_analysis_path = f"{folder_path}/comments_analysis.csv"
                                        result_df.to_csv(video_analysis_path, index=False)
                                        console.print(">>>> Analise de video [green]completada[/] -> Salvo em: '"+video_analysis_path+"'")
                                        month_data = pd.concat([month_data, result_df], ignore_index=True)
                                        videos_coletados += 1
                                        comentarios_coletados += result_df.loc[0,'comments_total']

                            month_csv_path = base_dir +f"/{ytb_folder}/{year_folder}/{month_folder}/{month_folder}_comments_analysis.csv"
                            month_data.to_csv(month_csv_path, index=False)
                            year_data = pd.concat([year_data, month_data],ignore_index=True)
                            console.print(">>> Analise [cyan]"+month_folder+"[/] [green]completada[/] -> Salvo em: '"+month_csv_path+"'")

                    year_csv_path = base_dir +f"/{ytb_folder}/{year_folder}/{year_folder}_comments_analysis.csv"
                    year_data.to_csv(year_csv_path, index=False)
                    ytb_data = pd.concat([ytb_data, year_data],ignore_index=True)
                    console.print(">> Analise [cyan]"+year_folder+"[/] [green]completada[/] -> Salvo em: '"+year_csv_path+"'")
            
            atualizar_video_comentarios_coletados(ytb_folder, videos_coletados, comentarios_coletados)
            ytb_csv_path = f"{base_dir}/{ytb_folder}/{ytb_folder}_comments_analysis.csv"
            ytb_data.to_csv(ytb_csv_path, index=False)
            console.print("> Analise [cyan]"+ytb_folder+"[/] [green]completada[/] -> Salvo em: '"+ytb_csv_path+"'")
            folder_path_graph = f"{base_dir}/{ytb_folder}"
            make_graph_neg_pos_comments(ytb_csv_path,folder_path_graph,ytb_folder)
            
if __name__ == "__main__":
    main()