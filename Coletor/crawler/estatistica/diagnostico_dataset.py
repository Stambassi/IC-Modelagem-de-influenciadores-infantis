import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Configurações de layout para padrão acadêmico
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

BASE_DIR = Path("files")

# Mapeamento fornecido
MAP_CATEGORIA_YOUTUBERS = {
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
    'TazerCraft': 'Minecraft'
}

'''
    Convert ISO 8601 duration (e.g., 'PT1H2M3S', 'PT18M24S', 'PT45S') into total seconds.
    Supports hours, minutes, and seconds.
'''
def iso_duration_to_seconds(duration: str) -> int:
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.fullmatch(duration)
    
    if not match:
        raise ValueError(f"Invalid duration format: {duration}")
    
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

'''
    Função auxiliar para ler a duração do vídeo
'''
def extrair_duracao(caminho_info):
    try:
        df = pd.read_csv(caminho_info)
        # Assumindo que a coluna se chame 'duration' e esteja em segundos.
        # Caso esteja no formato do YouTube (PT10M), será necessário um parse extra aqui.
        if 'duration' in df.columns:
            return iso_duration_to_seconds(df['duration'].iloc[0])
    except:
        pass
    return 0.0

def compilar_dados():
    linhas_tabela = []

    for youtuber, ambiente in MAP_CATEGORIA_YOUTUBERS.items():
        caminho_youtuber = BASE_DIR / youtuber
        
        videos_count = 0
        duracoes = []
        svts_por_video = []

        if caminho_youtuber.exists():
            for root, dirs, files in os.walk(caminho_youtuber):
                if "tiras_video.csv" in files:
                    caminho_pasta = Path(root)
                    
                    # 1. Conta as SVTs (linhas do tiras_video.csv)
                    try:
                        df_tiras = pd.read_csv(caminho_pasta / "tiras_video.csv")
                        total_svts = len(df_tiras)
                        svts_por_video.append(total_svts)
                    except:
                        continue
                        
                    # 2. Busca a duração no videos_info.csv
                    if "videos_info.csv" in files:
                        duracao = extrair_duracao(caminho_pasta / "videos_info.csv")
                        duracoes.append(duracao)
                        
                    videos_count += 1

        # Consolida as métricas do Youtuber
        if videos_count > 0:
            linhas_tabela.append({
                'Creator Name': youtuber,
                'Game Environment': ambiente,
                'Total Videos': videos_count,
                'Total Video Duration': np.sum(duracoes),
                'Avg. Video Duration': np.mean(duracoes),
                'Std. Video Duration': np.std(duracoes) if len(duracoes) > 1 else 0,
                'Total SVTs': np.sum(svts_por_video),
                'Avg. SVTs': np.mean(svts_por_video),
                'Std. SVTs': np.std(svts_por_video) if len(svts_por_video) > 1 else 0
            })

    df = pd.DataFrame(linhas_tabela)
    return df

def gerar_imagem_tabela(df, nome_saida="estatistica/tabela_diagnostico_dataset.png"):
    # Calcula a linha de Total
    total_row = {
        'Creator Name': 'Total',
        'Game Environment': '-',
        'Total Videos': df['Total Videos'].sum(),
        'Total Video Duration': df['Total Video Duration'].sum(),
        'Avg. Video Duration': df['Avg. Video Duration'].mean(), 
        'Std. Video Duration': df['Std. Video Duration'].mean(),
        'Total SVTs': df['Total SVTs'].sum(),
        'Avg. SVTs': df['Avg. SVTs'].mean(),
        'Std. SVTs': df['Std. SVTs'].mean()
    }
    
    # Adiciona a linha ao DataFrame
    df_final = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Conversão de segundos para minutos e renomeação
    cols_duracao = ['Total Video Duration', 'Avg. Video Duration', 'Std. Video Duration']
    df_final[cols_duracao] = df_final[cols_duracao] / 60.0
    
    df_final.rename(columns={
        'Total Video Duration': 'Total Video Duration (min)',
        'Avg. Video Duration': 'Avg. Video Duration (min)',
        'Std. Video Duration': 'Std. Video Duration (min)'
    }, inplace=True)
    
    # Arredondamento para estética acadêmica (2 casas decimais)
    cols_float = ['Total Video Duration (min)', 'Avg. Video Duration (min)', 'Std. Video Duration (min)', 'Avg. SVTs', 'Std. SVTs']
    df_final[cols_float] = df_final[cols_float].round(2)
    
    # Cria a figura (Aumentei a largura base para 16 para dar mais espaço)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300) 
    ax.axis('off')
    ax.axis('tight')
    
    # Desenha a tabela
    tabela = ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc='center', loc='center')
    
    # Customização de tipografia e cores
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.8) # Espaçamento vertical das células
    
    # Ajuste automático da largura das colunas baseado no conteúdo
    tabela.auto_set_column_width(col=list(range(len(df_final.columns))))
    
    for (row, col), cell in tabela.get_celld().items():
        # Estilo do Cabeçalho
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50') # Azul escuro acadêmico
        # Estilo da linha de Total (Última linha)
        elif row == len(df_final):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8F8F5') # Verde claro sutil para destaque
            
    plt.tight_layout()
    plt.savefig(nome_saida, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"[SUCESSO] Tabela gerada e salva como '{nome_saida}' na raiz do projeto!")
    
    df_final.to_csv("estatistica/tabela_diagnostico_dataset.csv", index=False)

if __name__ == "__main__":
    print("Iniciando varredura do dataset...")
    df_dados = compilar_dados()
    if not df_dados.empty:
        gerar_imagem_tabela(df_dados)
    else:
        print("[ERRO] Nenhum dado encontrado. Verifique o caminho da base de dados.")