import os
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações de layout para padrão acadêmico (Matplotlib)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# 1. PARAMETRIZAÇÃO DA ESCALA E VOLUME
# TIPO_ESCALA = 'likert' # Opções válidas: 'likert', 'binario', 'ternario'
TIPO_ESCALA = 'binario'
# TIPO_ESCALA = 'ternario'

TOTAL_AMOSTRA = 600

# 2. Configuração de Voluntários (Overlap Exato de 2 por SVT)
CLASSIFICADORES = ['Avaliador_A', 'Avaliador_B', 'Avaliador_C', 'Avaliador_D']

# 3. Pesos da Amostragem para o Label Studio (Target de Validação)
PESOS_AMOSTRAGEM = {
    'NT': 0.20, # 20% Neutro (120 SVTs)
    'GZ': 0.60, # 60% Zona Cinzenta (360 SVTs)
    'T':  0.20  # 20% Tóxico (120 SVTs)
}

# 4. Limiares (Thresholds) dos Modelos de IA
THRESHOLDS_DETOXIFY = {
    'max_nt': 0.20,
    'min_t': 0.80
}
THRESHOLDS_PERSPECTIVE = {
    'max_nt': 0.20,
    'min_t': 0.40
}

# Configurações de diretório e mapeamento
BASE_DIR = Path("files")
ANOTACAO_DIR = Path("anotacao/dados")
ARQUIVO_TOTAL = ANOTACAO_DIR / "total_tiras.csv"
ARQUIVO_AMOSTRA_MASTER = ANOTACAO_DIR / TIPO_ESCALA / f"dados_amostrais_master_{TIPO_ESCALA}.csv"

MAP_CATEGORIA_YOUTUBERS = {
    'Julia MineGirl': 'Roblox', 'Papile': 'Roblox', 'Tex HS': 'Roblox',
    'Amy Scarlet': 'Roblox', 'Luluca Games': 'Roblox', 'meu nome é david': 'Roblox',
    'Lokis': 'Roblox', 'Robin Hood Gamer': 'Minecraft', 'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft', 'Athos': 'Minecraft', 'JP Plays': 'Minecraft',
    'Marcelodrv': 'Minecraft', 'Geleia': 'Minecraft', 'Kass e KR': 'Minecraft',
    'TazerCraft': 'Minecraft'
}

# Funções de processamento e amostragem

'''
    Varre o diretório de dados estruturados e cria um arquivo mestre 
    'total_tiras.csv' contendo todas as SVTs processadas, caso ele não exista.
'''
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

'''
    Classifica um score numérico individual em classes (NT, GZ, T) 
    baseado nos limiares fornecidos por parâmetro.
'''
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

'''
    Gera as classificações de ambos os modelos (Detoxify e Perspective API)
    e filtra o DataFrame mantendo apenas as SVTs onde há consenso exato.
'''
def filtrar_por_consenso(df):
    print("\n--- Calculando Consenso entre Modelos ---")
    df['class_detoxify'] = df['toxicity'].apply(lambda x: classificar_score(x, THRESHOLDS_DETOXIFY))
    df['class_perspective'] = df['p_toxicity'].apply(lambda x: classificar_score(x, THRESHOLDS_PERSPECTIVE))
    
    df_consenso = df[df['class_detoxify'] == df['class_perspective']].copy()
    df_consenso['classificacao_ia'] = df_consenso['class_detoxify']
    
    print(f"Total de SVTs original: {len(df)}")
    print(f"Total de SVTs com consenso: {len(df_consenso)}")
    return df_consenso

'''
    Gera uma tabela formatada (PNG e CSV) mostrando a quantidade e proporção 
    de SVTs de cada categoria, herdando o sufixo da escala atual.
'''
def gerar_tabela_distribuicao_classes(df_original, df_consenso):
    print("\n--- Gerando Tabela de Diagnóstico de Classes ---")
    orig_counts = df_original.groupby('Creator Name').size().rename('Original SVTs')
    dist = df_consenso.groupby(['Creator Name', 'classificacao_ia']).size().unstack(fill_value=0)
    
    for col in ['NT', 'GZ', 'T']:
        if col not in dist.columns: dist[col] = 0
            
    dist = dist.join(orig_counts)
    dist = dist[['Original SVTs', 'NT', 'GZ', 'T']]
    dist['Consensus SVTs'] = dist[['NT', 'GZ', 'T']].sum(axis=1)
    
    dist['Consensus (%)'] = (dist['Consensus SVTs'] / dist['Original SVTs'] * 100).round(2)
    dist['NT (%)'] = (dist['NT'] / dist['Consensus SVTs'] * 100).round(2)
    dist['GZ (%)'] = (dist['GZ'] / dist['Consensus SVTs'] * 100).round(2)
    dist['T (%)'] = (dist['T'] / dist['Consensus SVTs'] * 100).round(2)
    
    dist = dist.reset_index()
    
    total_orig, total_cons = dist['Original SVTs'].sum(), dist['Consensus SVTs'].sum()
    total_nt, total_gz, total_t = dist['NT'].sum(), dist['GZ'].sum(), dist['T'].sum()
    
    total_row = {
        'Creator Name': 'Total / Average',
        'Original SVTs': total_orig, 'Consensus SVTs': total_cons,
        'Consensus (%)': round((total_cons / total_orig) * 100, 2) if total_orig else 0,
        'NT': total_nt, 'NT (%)': round((total_nt / total_cons) * 100, 2) if total_cons else 0,
        'GZ': total_gz, 'GZ (%)': round((total_gz / total_cons) * 100, 2) if total_cons else 0,
        'T': total_t, 'T (%)': round((total_t / total_cons) * 100, 2) if total_cons else 0
    }
    
    cols_order = ['Creator Name', 'Original SVTs', 'Consensus SVTs', 'Consensus (%)', 'NT', 'NT (%)', 'GZ', 'GZ (%)', 'T', 'T (%)']
    df_final = pd.concat([dist, pd.DataFrame([total_row])], ignore_index=True)[cols_order]
    
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300) 
    ax.axis('off'); ax.axis('tight')
    tabela = ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc='center', loc='center')
    tabela.auto_set_font_size(False); tabela.set_fontsize(10); tabela.scale(1.2, 1.8)
    tabela.auto_set_column_width(col=list(range(len(df_final.columns))))
    
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        elif row == len(df_final):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8F8F5')
            
    plt.tight_layout()
    caminho_png = ANOTACAO_DIR / f"tabela_distribuicao.png"
    caminho_csv = ANOTACAO_DIR / f"tabela_distribuicao.csv"
    plt.savefig(caminho_png, bbox_inches='tight', transparent=False, facecolor='white')
    df_final.to_csv(caminho_csv, index=False)

'''
    Distribui as cotas de forma exatamente igualitária. Se faltar dados em um Youtuber,
    redistribui a falta para os demais, garantindo que a soma bata perfeitamente o 'alvo_total'.
'''
def calcular_cotas_exatas(df_classe, alvo_total):
    criadores = sorted(df_classe['Creator Name'].unique())
    disponivel = {c: len(df_classe[df_classe['Creator Name'] == c]) for c in criadores}
    cotas = {c: 0 for c in criadores}
    
    restante = alvo_total
    ativos = [c for c in criadores if disponivel[c] > 0]
    
    while restante > 0 and ativos:
        cota_base = restante // len(ativos)
        extra = restante % len(ativos)
        
        ativos.sort() # Garante reprodutibilidade da distribuição do 'extra'
        alocacoes_rodada = {}
        
        for i, c in enumerate(ativos):
            necessidade = cota_base + (1 if i < extra else 0)
            pode_pegar = min(necessidade, disponivel[c] - cotas[c])
            alocacoes_rodada[c] = pode_pegar
            
        for c, qtd in alocacoes_rodada.items():
            cotas[c] += qtd
            restante -= qtd
            if cotas[c] == disponivel[c]:
                ativos.remove(c)
                
    return cotas

'''
    Gera a amostra com 600 SVTs dividida entre as 3 classes.
    Embaralha os resultados para Label Studio, cria os textos parametrizados 
    (ocultando IA em binário/ternário) e designa os avaliadores.
'''
def gerar_amostra_estratificada(df_consenso):
    print(f"\n--- Iniciando Amostragem Estratificada [{TIPO_ESCALA.upper()}] (Alvo: {TOTAL_AMOSTRA} SVTs) ---")
    amostras_finais = []
    
    # 1. Estratificação e Algoritmo de Cotas Exatas
    for classe, peso in PESOS_AMOSTRAGEM.items():
        alvo_bucket = int(TOTAL_AMOSTRA * peso)
        print(f"\n[Classe {classe}] Alvo Total: {alvo_bucket}")
        df_classe = df_consenso[df_consenso['classificacao_ia'] == classe]
        
        # Calcula as cotas de forma determinística
        cotas_calculadas = calcular_cotas_exatas(df_classe, alvo_bucket)
        
        for criador, cota in cotas_calculadas.items():
            if cota > 0:
                df_criador = df_classe[df_classe['Creator Name'] == criador]
                # Random state garante que os mesmos SVTs sejam sorteados caso rode novamente
                amostra = df_criador.sample(n=cota, random_state=42)
                amostras_finais.append(amostra)
                
    # Concatena todas as 600 SVTs e embaralha (livre de viés de ordenação)
    df_amostra_final = pd.concat(amostras_finais).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Formatação da String de Exibição
    base_metadados = (
        "🎬 Canal: " + df_amostra_final['Creator Name'] + " | 🎥 Vídeo: " + df_amostra_final['Video Name'] + "\n" +
        "⏱️ Segmento/Index: " + df_amostra_final['index'].astype(str) + "\n"
    )
    base_texto = "──────────────────────────────────────────────────\nFala transcrita:\n\"" + df_amostra_final['tiras'] + "\""

    if TIPO_ESCALA == 'likert':
        metadados_ia = (
            "🤖 Consenso de IA: " + df_amostra_final['classificacao_ia'] + 
            " (Detoxify: " + df_amostra_final['toxicity'].astype(float).round(3).astype(str) + 
            " | Perspective: " + df_amostra_final['p_toxicity'].astype(float).round(3).astype(str) + ")\n"
        )
        df_amostra_final['texto_exibicao'] = base_metadados + metadados_ia + base_texto
    elif TIPO_ESCALA in ['binario', 'ternario']:
        # Teste cego: Sem vazamento da classificação de IA
        df_amostra_final['texto_exibicao'] = base_metadados + base_texto

    # 3. Distribuição Combinatória dos Avaliadores (2 por SVT)
    # 4 avaliadores geram 6 pares exatos. 600 SVTs / 6 pares = 100 SVTs por par.
    pares_avaliadores = list(itertools.combinations(CLASSIFICADORES, 2))
    tamanho_chunk = len(df_amostra_final) // len(pares_avaliadores)
    
    anotador_1, anotador_2 = [], []
    for i in range(len(df_amostra_final)):
        idx_par = (i // tamanho_chunk) % len(pares_avaliadores)
        anotador_1.append(pares_avaliadores[idx_par][0])
        anotador_2.append(pares_avaliadores[idx_par][1])
        
    df_amostra_final['Anotador_1'] = anotador_1
    df_amostra_final['Anotador_2'] = anotador_2

    print(f"\n[SUCESSO] Amostra gerada: {len(df_amostra_final)} SVTs (300 tarefas exatas para cada classificador).")
    return df_amostra_final

'''
    Lê o DataFrame mestre e gera arquivos CSV separados para cada 
    classificador contendo exclusivamente a carga de trabalho de 300 SVTs dele.
'''
def gerar_arquivos_individuais_avaliadores(df_master):
    print("\n--- Gerando Carga de Trabalho Individual ---")
    for avaliador in CLASSIFICADORES:
        # Filtra as linhas onde o avaliador aparece na coluna 1 ou na coluna 2
        df_individual = df_master[(df_master['Anotador_1'] == avaliador) | (df_master['Anotador_2'] == avaliador)]
        
        caminho_indiv = ANOTACAO_DIR / TIPO_ESCALA
        caminho_indiv.mkdir(parents=True, exist_ok=True)

        df_individual.to_csv(caminho_indiv / f"dados_amostrais_{TIPO_ESCALA}_{avaliador}.csv", index=False)
        print(f" └── {avaliador}: {len(df_individual)} SVTs prontas ({caminho_indiv.name})")

# Fluxo principal
if __name__ == "__main__":
    df_completo = compilar_dataset_total()
    df_filtrado_consenso = filtrar_por_consenso(df_completo)
    gerar_tabela_distribuicao_classes(df_completo, df_filtrado_consenso)
    
    # Gera a base com as designações de avaliadores
    df_master_label_studio = gerar_amostra_estratificada(df_filtrado_consenso)
    
    # Salva o arquivo Master
    (ANOTACAO_DIR / TIPO_ESCALA).mkdir(parents=True, exist_ok=True)
    df_master_label_studio.to_csv(ARQUIVO_AMOSTRA_MASTER, index=False)
    print(f"\n[FINALIZADO] Arquivo Master salvo: {ARQUIVO_AMOSTRA_MASTER}")
    
    # Opcional (porém super recomendado): Exporta o arquivo segmentado por voluntário
    gerar_arquivos_individuais_avaliadores(df_master_label_studio)