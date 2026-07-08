import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações de layout para padrão acadêmico (Matplotlib)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# CONFIGURAÇÕES METODOLÓGICAS (Parâmetros de Fácil Alteração)

# 1. PARAMETRIZAÇÃO DA ESCALA DE ANOTAÇÃO
# Opções válidas: 'likert', 'binario', 'ternario'
# TIPO_ESCALA = 'likert'
# TIPO_ESCALA = 'binario'
TIPO_ESCALA = 'ternario'

# 2. Pesos da Amostragem para o Label Studio (Target de Validação)
PESOS_AMOSTRAGEM = {
    'NT': 0.20, # 20% Neutro
    'GZ': 0.60, # 60% Zona Cinzenta
    'T':  0.20  # 20% Tóxico
}

# 3. Limiares (Thresholds) dos Modelos de IA
THRESHOLDS_DETOXIFY = {
    'max_nt': 0.20,
    'min_t': 0.80
}

THRESHOLDS_PERSPECTIVE = {
    'max_nt': 0.20,
    'min_t': 0.40
}

# CONFIGURAÇÕES DE DIRETÓRIOS E MAPEAMENTO
BASE_DIR = Path("files")
ANOTACAO_DIR = Path("anotacao/dados")
ARQUIVO_TOTAL = ANOTACAO_DIR / "total_tiras.csv"

# O nome do arquivo de amostra agora é dinâmico com base na escala escolhida
ARQUIVO_AMOSTRA = ANOTACAO_DIR / f"dados_amostrais_{TIPO_ESCALA}.csv"

MAP_CATEGORIA_YOUTUBERS = {
    'Julia MineGirl': 'Roblox', 'Papile': 'Roblox', 'Tex HS': 'Roblox',
    'Amy Scarlet': 'Roblox', 'Luluca Games': 'Roblox', 'meu nome é david': 'Roblox',
    'Lokis': 'Roblox', 'Robin Hood Gamer': 'Minecraft', 'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft', 'Athos': 'Minecraft', 'JP Plays': 'Minecraft',
    'Marcelodrv': 'Minecraft', 'Geleia': 'Minecraft', 'Kass e KR': 'Minecraft',
    'TazerCraft': 'Minecraft'
}

# FUNÇÕES DE PROCESSAMENTO

"""
    Varre o diretório de dados estruturados e cria um arquivo mestre 
    'total_tiras.csv' contendo todas as SVTs processadas, caso ele não exista.
"""
def compilar_dataset_total():
    if ARQUIVO_TOTAL.exists():
        print(f"[INFO] Arquivo mestre já existe em {ARQUIVO_TOTAL}. Lendo dados...")
        return pd.read_csv(ARQUIVO_TOTAL)
    
    print("[INFO] Arquivo mestre não encontrado. Compilando SVTs. Isso pode levar alguns segundos...")
    ANOTACAO_DIR.mkdir(parents=True, exist_ok=True)
    
    lista_dfs = []
    
    for youtuber, ambiente in MAP_CATEGORIA_YOUTUBERS.items():
        caminho_youtuber = BASE_DIR / youtuber
        
        if caminho_youtuber.exists():
            for root, dirs, files in os.walk(caminho_youtuber):
                if "tiras_video.csv" in files:
                    caminho_csv = Path(root) / "tiras_video.csv"
                    nome_video = Path(root).name
                    
                    try:
                        df_temp = pd.read_csv(caminho_csv)
                        df_temp['Creator Name'] = youtuber
                        df_temp['Game Environment'] = ambiente
                        df_temp['Video Name'] = nome_video
                        lista_dfs.append(df_temp)
                    except Exception as e:
                        print(f"[ERRO] Falha ao ler {caminho_csv}: {e}")
                        
    if not lista_dfs:
        raise ValueError("Nenhum dado encontrado! Verifique a pasta 'files'.")
        
    df_total = pd.concat(lista_dfs, ignore_index=True)
    df_total.to_csv(ARQUIVO_TOTAL, index=False)
    print(f"[SUCESSO] Dataset mestre criado com {len(df_total)} SVTs totais em {ARQUIVO_TOTAL}.")
    
    return df_total


"""
    Classifica um score numérico individual em classes (NT, GZ, T) baseado nos limiares fornecidos por parâmetro
"""
def classificar_score(score, thresholds):
    try:
        val = float(score)
    except:
        return 'ERRO'
        
    if val < thresholds['max_nt']:
        return 'NT'
    elif val >= thresholds['min_t']:
        return 'T'
    else:
        return 'GZ'


"""
    Gera as classificações de ambos os modelos (Detoxify e Perspective API)
    e filtra o DataFrame mantendo apenas as SVTs onde há consenso exato.
"""
def filtrar_por_consenso(df):
    print("\n--- Calculando Consenso entre Modelos ---")
    
    df['class_detoxify'] = df['toxicity'].apply(lambda x: classificar_score(x, THRESHOLDS_DETOXIFY))
    df['class_perspective'] = df['p_toxicity'].apply(lambda x: classificar_score(x, THRESHOLDS_PERSPECTIVE))
    
    # Filtra mantendo apenas concordância estrita
    df_consenso = df[df['class_detoxify'] == df['class_perspective']].copy()
    df_consenso['classificacao_ia'] = df_consenso['class_detoxify']
    
    print(f"Total de SVTs original: {len(df)}")
    print(f"Total de SVTs com consenso (Detoxify == Perspective): {len(df_consenso)}")
    print(f"SVTs descartadas por discordância: {len(df) - len(df_consenso)}")
    
    return df_consenso

"""
    Gera uma tabela formatada (PNG e CSV) mostrando a quantidade e proporção 
    de SVTs de cada categoria para cada youtuber. Os arquivos herdam o sufixo da escala.
"""
def gerar_tabela_distribuicao_classes(df_consenso):
    print("\n--- Gerando Tabela de Diagnóstico de Classes ---")
    
    # 1. Agrupa os dados contando as ocorrências
    dist = df_consenso.groupby(['Creator Name', 'classificacao_ia']).size().unstack(fill_value=0)
    
    for col in ['NT', 'GZ', 'T']:
        if col not in dist.columns:
            dist[col] = 0
            
    # Ordena colunas e soma totais
    dist = dist[['NT', 'GZ', 'T']]
    dist['Total SVTs'] = dist.sum(axis=1)
    
    # 2. Calcula as porcentagens
    dist['NT (%)'] = (dist['NT'] / dist['Total SVTs'] * 100).round(2)
    dist['GZ (%)'] = (dist['GZ'] / dist['Total SVTs'] * 100).round(2)
    dist['T (%)'] = (dist['T'] / dist['Total SVTs'] * 100).round(2)
    
    dist = dist.reset_index()
    
    # 3. Calcula a linha consolidadora
    total_nt, total_gz, total_t = dist['NT'].sum(), dist['GZ'].sum(), dist['T'].sum()
    total_svts = dist['Total SVTs'].sum()
    
    total_row = {
        'Creator Name': 'Total / Average',
        'Total SVTs': total_svts,
        'NT': total_nt,
        'NT (%)': round((total_nt / total_svts) * 100, 2) if total_svts else 0,
        'GZ': total_gz,
        'GZ (%)': round((total_gz / total_svts) * 100, 2) if total_svts else 0,
        'T': total_t,
        'T (%)': round((total_t / total_svts) * 100, 2) if total_svts else 0
    }
    
    cols_order = ['Creator Name', 'Total SVTs', 'NT', 'NT (%)', 'GZ', 'GZ (%)', 'T', 'T (%)']
    df_final = pd.concat([dist, pd.DataFrame([total_row])], ignore_index=True)
    df_final = df_final[cols_order]
    
    # 4. Configuração visual e salvamento da tabela
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300) 
    ax.axis('off')
    ax.axis('tight')
    
    tabela = ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc='center', loc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.8)
    tabela.auto_set_column_width(col=list(range(len(df_final.columns))))
    
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        elif row == len(df_final):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8F8F5')
            
    plt.tight_layout()
    
    caminho_png = ANOTACAO_DIR / "tabela_distribuicao_classes.png"
    caminho_csv = ANOTACAO_DIR / "tabela_distribuicao_classes.csv"
    
    plt.savefig(caminho_png, bbox_inches='tight', transparent=False, facecolor='white')
    df_final.to_csv(caminho_csv, index=False)
    print(f"[SUCESSO] Tabela gerada em:\n └── {caminho_png}\n └── {caminho_csv}")


"""
    Gera a amostra garantindo a representatividade igualitária entre os Youtubers.
    Prepara a string de exibição ('texto_exibicao') de forma condicional, ocultando 
    o rótulo da IA para escalas binárias e ternárias para evitar viés do anotador.
"""
def gerar_amostra_estratificada(df_consenso, total_amostra_desejada=1000):
    print(f"\n--- Iniciando Amostragem Estratificada [{TIPO_ESCALA.upper()}] (Alvo: {total_amostra_desejada} SVTs) ---")
    
    lista_criadores = df_consenso['Creator Name'].unique()
    num_criadores = len(lista_criadores)
    
    amostras_finais = []
    
    # 1. Processo de Amostragem Estratificada
    for classe, peso in PESOS_AMOSTRAGEM.items():
        alvo_bucket = int(total_amostra_desejada * peso)
        alvo_por_criador = math.ceil(alvo_bucket / num_criadores)
        
        print(f"\n[Classe {classe}] Alvo Total: {alvo_bucket} | Cota por Criador: {alvo_por_criador}")
        df_classe = df_consenso[df_consenso['classificacao_ia'] == classe]
        
        for criador in lista_criadores:
            df_criador = df_classe[df_classe['Creator Name'] == criador]
            disponivel = len(df_criador)
            
            n_amostras = min(alvo_por_criador, disponivel)
            
            if n_amostras < alvo_por_criador:
                print(f"  └── Aviso: '{criador}' tem apenas {disponivel} SVTs em consenso para {classe} (Faltam {alvo_por_criador - disponivel}).")
                
            if n_amostras > 0:
                amostra = df_criador.sample(n=n_amostras, random_state=42)
                amostras_finais.append(amostra)
                
    df_amostra_final = pd.concat(amostras_finais).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Formatação da String de Exibição Baseada no Tipo de Escala
    base_metadados = (
        "🎬 Canal: " + df_amostra_final['Creator Name'] + " | 🎥 Vídeo: " + df_amostra_final['Video Name'] + "\n" +
        "⏱️ Segmento/Index: " + df_amostra_final['index'].astype(str) + "\n"
    )
    
    base_texto = (
        "──────────────────────────────────────────────────\n" +
        "Fala transcrita:\n\"" + df_amostra_final['tiras'] + "\""
    )

    if TIPO_ESCALA == 'likert':
        # Exibe as predições e os scores para que o humano avalie a performance da máquina
        metadados_ia = (
            "🤖 Consenso de IA: " + df_amostra_final['classificacao_ia'] + 
            " (Detoxify: " + df_amostra_final['toxicity'].astype(float).round(3).astype(str) + 
            " | Perspective: " + df_amostra_final['p_toxicity'].astype(float).round(3).astype(str) + ")\n"
        )
        df_amostra_final['texto_exibicao'] = base_metadados + metadados_ia + base_texto
        
    elif TIPO_ESCALA in ['binario', 'ternario']:
        # Teste cego: Oculta qualquer informação de IA para evitar viés de confirmação
        df_amostra_final['texto_exibicao'] = base_metadados + base_texto
        
    else:
        raise ValueError(f"Escala '{TIPO_ESCALA}' não é suportada. Use 'likert', 'binario' ou 'ternario'.")

    print(f"\n[SUCESSO] Amostra gerada com {len(df_amostra_final)} SVTs no formato {TIPO_ESCALA}.")
    return df_amostra_final


if __name__ == "__main__":
    # 1. Carrega ou cria a base completa
    df_completo = compilar_dataset_total()
    
    # 2. Aplica as regras de consenso entre Detoxify e Perspective API
    df_filtrado_consenso = filtrar_por_consenso(df_completo)
    
    # 3. Gera a tabela estatística das classes pós-consenso
    gerar_tabela_distribuicao_classes(df_filtrado_consenso)
    
    # 4. Gera a amostra formatada para a validação humana
    df_label_studio = gerar_amostra_estratificada(df_filtrado_consenso, total_amostra_desejada=1000)
    
    # 5. Salva o resultado no arquivo final com o sufixo correspondente
    df_label_studio.to_csv(ARQUIVO_AMOSTRA, index=False)
    print(f"\n[FINALIZADO] Arquivo de anotação salvo em: {ARQUIVO_AMOSTRA}")