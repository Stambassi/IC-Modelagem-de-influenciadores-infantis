import os
import pandas as pd
import numpy as np
import krippendorff
from irrCAC.raw import CAC
from sklearn.metrics import cohen_kappa_score
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# =====================================================================
# CONFIGURAÇÕES METODOLÓGICAS E DE DIRETÓRIO
# =====================================================================

# Configurações de layout para padrão acadêmico (Matplotlib)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Diretórios de Entrada e Saída
DIRETORIO_DADOS = "anotacao/dados_exportados"
DIRETORIO_RELATORIOS = "anotacao/relatorios"

# Nomes das colunas originais
COLUNA_INDEX = "index"             
COLUNA_VIDEO = "Video Name"        
COLUNA_ANOTADOR = "annotator"      
COLUNA_ROTULO = "binary_scale"     
COLUNA_TOXICITY = "toxicity"       # Detoxify raw score
COLUNA_PTOXICITY = "p_toxicity"    # Perspective API raw score
COLUNA_TEXTO = "tiras"

# Nova Chave Primária Global que será gerada dinamicamente
COLUNA_ID_SVT = "svt_uid"

# Labels exatos das classes para mapeamento
LABEL_NT = 'Não Toxico'
LABEL_GZ = 'Zona Cinzenta'
LABEL_T = 'Tóxico'

# =====================================================================
# FUNÇÕES DE PREPARAÇÃO DE DADOS
# =====================================================================

"""
    Lê todos os CSVs do diretório e empilha em um único DataFrame.
    Retém os scores originais das IAs e o texto (tiras) para permitir 
    o Threshold Tuning e a Análise Qualitativa.
"""
def carregar_e_preparar_dados(diretorio):
    arquivos = [f for f in os.listdir(diretorio) if f.endswith('.csv')]
    
    if not arquivos:
        raise ValueError(f"[ERRO] Nenhum arquivo CSV encontrado em {diretorio}.")
    
    dfs = []
    for arquivo in arquivos:
        caminho = os.path.join(diretorio, arquivo)
        df = pd.read_csv(caminho)
        
        colunas_necessarias_originais = [
            COLUNA_INDEX, COLUNA_VIDEO, COLUNA_ANOTADOR, 
            COLUNA_ROTULO, COLUNA_TOXICITY, COLUNA_PTOXICITY, COLUNA_TEXTO
        ]
        
        for col in colunas_necessarias_originais:
            if col not in df.columns:
                raise KeyError(f"[ERRO] A coluna '{col}' não existe no arquivo {arquivo}.")
        
        # Cria a Chave Primária Global
        df[COLUNA_ID_SVT] = (df[COLUNA_VIDEO].str.extract(r"\[(.*?)\]", expand=False) + " - idx:" + df[COLUNA_INDEX].astype(str))
        df[COLUNA_ANOTADOR] = df[COLUNA_ANOTADOR].astype(str)
        
        # Converte os scores de IA para numérico garantindo a matemática
        df[COLUNA_TOXICITY] = pd.to_numeric(df[COLUNA_TOXICITY], errors='coerce')
        df[COLUNA_PTOXICITY] = pd.to_numeric(df[COLUNA_PTOXICITY], errors='coerce')
        
        colunas_finais = [
            COLUNA_ID_SVT, COLUNA_ANOTADOR, COLUNA_ROTULO, 
            COLUNA_TOXICITY, COLUNA_PTOXICITY, COLUNA_TEXTO
        ]
        dfs.append(df[colunas_finais])
        
    df_consolidado = pd.concat(dfs, ignore_index=True)
    df_consolidado = df_consolidado.dropna(subset=[COLUNA_ROTULO, COLUNA_TOXICITY, COLUNA_PTOXICITY])
    
    return df_consolidado

def criar_matriz_esparsa(df):
    matriz = df.pivot(index=COLUNA_ID_SVT, columns=COLUNA_ANOTADOR, values=COLUNA_ROTULO)
    return matriz

# =====================================================================
# FUNÇÕES DE CÁLCULO DE MÉTRICAS GLOBAIS (IRR)
# =====================================================================

def calcular_kappa_cohen_medio(matriz):
    anotadores = matriz.columns
    pares = list(itertools.combinations(anotadores, 2))
    kappas = []
    
    for a1, a2 in pares:
        sobreposicao = matriz[[a1, a2]].dropna()
        if len(sobreposicao) > 0:
            k = cohen_kappa_score(sobreposicao[a1], sobreposicao[a2])
            kappas.append(k)
            
    if not kappas: return 0.0
    return np.mean(kappas)

def reportar_metricas(df_raw, matriz_pivot):
    print("\n" + "="*50)
    print(" RELATÓRIO DE CONCORDÂNCIA GLOBAL (IRR)")
    print("="*50)
    
    total_svts = len(matriz_pivot)
    anotadores_unicos = matriz_pivot.columns.tolist()
    
    print(f"Total de SVTs únicas avaliadas: {total_svts}")
    print(f"Anotadores presentes na base ({len(anotadores_unicos)}): {', '.join(map(str, anotadores_unicos))}\n")

    cohen_medio = calcular_kappa_cohen_medio(matriz_pivot)
    print(f"➜ Kappa de Cohen (Média Pairwise): {cohen_medio:.4f}")

    cac = CAC(matriz_pivot)
    try:
        fleiss = cac.fleiss()['est']['coefficient_value']
        print(f"➜ Kappa de Fleiss: {fleiss:.4f}")
    except: print("➜ Kappa de Fleiss: [Não calculável]")

    try:
        gwet = cac.gwet()['est']['coefficient_value']
        print(f"➜ Gwet's AC1: {gwet:.4f}")
    except: print("➜ Gwet's AC1: [Não calculável]")

    try:
        matriz_numerica = matriz_pivot.apply(lambda x: pd.factorize(x)[0]).replace(-1, np.nan)
        dados_krip = matriz_numerica.T.values.tolist()
        alpha = krippendorff.alpha(reliability_data=dados_krip, level_of_measurement='nominal')
        print(f"➜ Alpha de Krippendorff: {alpha:.4f}")
    except: print("➜ Alpha de Krippendorff: [Não calculável]")
    print("="*50)

# =====================================================================
# THRESHOLD TUNING & VISUALIZAÇÕES
# =====================================================================

"""
    Executa um Grid Search varrendo valores de 0.05 a 0.95.
    Descobre os melhores pontos T1 e T2 para maximizar o Kappa de Cohen
    entre a IA e cada anotador humano, de forma isolada (Maximização por Perfil).
"""
def otimizar_thresholds_por_anotador(df_raw):
    print("\n--- Iniciando Threshold Tuning (Grid Search) por Perfil ---")
    anotadores = df_raw[COLUNA_ANOTADOR].unique()
    modelos = [COLUNA_TOXICITY, COLUNA_PTOXICITY]
    
    resultados = []
    
    # Range de busca (0.05 a 0.95, de 5 em 5)
    range_t1 = np.arange(0.05, 0.90, 0.05)
    
    for anotador in anotadores:
        df_anot = df_raw[df_raw[COLUNA_ANOTADOR] == anotador]
        y_true = df_anot[COLUNA_ROTULO]
        
        for modelo in modelos:
            best_k = -1.0
            best_t1, best_t2 = 0.0, 0.0
            
            # Otimização Combinatória de Thresholds
            for t1 in range_t1:
                range_t2 = np.arange(t1 + 0.05, 0.96, 0.05)
                for t2 in range_t2:
                    # Função rápida de classificação baseada nos thresholds atuais
                    def classificar(score):
                        if score < t1: return LABEL_NT
                        elif score >= t2: return LABEL_T
                        else: return LABEL_GZ
                        
                    y_pred = df_anot[modelo].apply(classificar)
                    k = cohen_kappa_score(y_true, y_pred)
                    
                    if k > best_k:
                        best_k = k
                        best_t1 = t1
                        best_t2 = t2
                        
            resultados.append({
                'Anotador': anotador,
                'Modelo': 'Detoxify' if modelo == COLUNA_TOXICITY else 'Perspective API',
                'T1 (NT->GZ)': round(best_t1, 2),
                'T2 (GZ->T)': round(best_t2, 2),
                'Max Kappa': round(best_k, 4)
            })
            
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(os.path.join(DIRETORIO_RELATORIOS, "threshold_tuning_resultados.csv"), index=False)
    print(" └── Tabela de otimização salva em: 'threshold_tuning_resultados.csv'")
    
    # Gera a visualização da Zona Cinzenta ideal para cada anotador
    plotar_thresholds_otimizados(df_resultados)
    
"""
    Cria um gráfico de barras empilhadas horizontais evidenciando
    qual é o tamanho da "Zona Cinzenta" que calibra a IA ao perfil do humano.
"""
def plotar_thresholds_otimizados(df_res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    nomes_modelos = ['Detoxify', 'Perspective API']
    
    for i, nome_mod in enumerate(nomes_modelos):
        ax = axes[i]
        df_mod = df_res[df_res['Modelo'] == nome_mod]
        
        anotadores = df_mod['Anotador'].tolist()
        t1s = df_mod['T1 (NT->GZ)'].tolist()
        t2s = df_mod['T2 (GZ->T)'].tolist()
        kappas = df_mod['Max Kappa'].tolist()
        y_pos = np.arange(len(anotadores))
        
        # Plota os 3 blocos (NT, GZ, T)
        ax.barh(y_pos, t1s, color='#A8E6CF', edgecolor='black', label='Não Tóxico (Score < T1)')
        ax.barh(y_pos, np.array(t2s) - np.array(t1s), left=t1s, color='#E0E0E0', edgecolor='black', label='Zona Cinzenta (T1 ≤ Score < T2)')
        ax.barh(y_pos, 1.0 - np.array(t2s), left=t2s, color='#FF8B94', edgecolor='black', label='Tóxico (Score ≥ T2)')
        
        # Adiciona o label mostrando qual foi o Kappa máximo alcançado
        for j, k in enumerate(kappas):
            ax.text(1.02, j, f"Max Kappa: {k:.2f}", va='center', fontsize=9, weight='bold')
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Anotador {a}" for a in anotadores])
        ax.set_xlim(0, 1.0)
        ax.set_title(nome_mod, weight='bold')
        ax.set_xlabel('Score IA (0.0 a 1.0)')
        
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(1.05, -0.15), ncol=3)
            
    plt.suptitle("Threshold Tuning: O Limiar de IA Ideal para cada Perfil Humano", weight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_RELATORIOS, "threshold_tuning_visualizacao.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" └── Gráfico da Zona Cinzenta ideal salvo em: 'threshold_tuning_visualizacao.png'\n")

"""
    Gera relatórios granulares para entender o viés dos anotadores, 
    a concordância por classe (NT, GZ, T) e o cruzamento par-a-par.
    (Versão completa com plotagem em alta resolução restaurada).
"""
def gerar_relatorios_estatisticos_visuais(df_raw, matriz_pivot):
    os.makedirs(DIRETORIO_RELATORIOS, exist_ok=True)
    anotadores = matriz_pivot.columns.tolist()
    
    print("--- Gerando Relatórios Visuais Base ---")
    
    # =================================================================
    # 1. Distribuição de Rótulos por Anotador (Gráfico de Viés)
    # =================================================================
    dist_anotador = df_raw.groupby([COLUNA_ANOTADOR, COLUNA_ROTULO]).size().unstack(fill_value=0)
    dist_anotador.to_csv(os.path.join(DIRETORIO_RELATORIOS, "distribuicao_por_anotador.csv"))
    
    # Plotagem do Gráfico de Barras Empilhadas
    ax = dist_anotador.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', edgecolor='black')
    plt.title('Distribuição de Rótulos por Anotador (Viés de Classificação)', pad=15, weight='bold')
    plt.ylabel('Total de SVTs Avaliadas')
    plt.xlabel('Anotador ID')
    plt.xticks(rotation=0)
    plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_RELATORIOS, "grafico_vies_anotadores.png"), dpi=300)
    plt.close()
    
    # =================================================================
    # 2. Análise de Concordância Pairwise (Heatmap de Kappa)
    # =================================================================
    n = len(anotadores)
    matriz_kappa = np.zeros((n, n))
    
    for i, a1 in enumerate(anotadores):
        for j, a2 in enumerate(anotadores):
            if i == j:
                matriz_kappa[i, j] = 1.0
            else:
                sobreposicao = matriz_pivot[[a1, a2]].dropna()
                if len(sobreposicao) > 0:
                    matriz_kappa[i, j] = cohen_kappa_score(sobreposicao[a1], sobreposicao[a2])
                else:
                    matriz_kappa[i, j] = np.nan
                    
    df_kappa = pd.DataFrame(matriz_kappa, index=anotadores, columns=anotadores)
    df_kappa.to_csv(os.path.join(DIRETORIO_RELATORIOS, "matriz_kappa_pairwise.csv"))
    
    # Plotagem do Heatmap de Kappa
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cax = ax.matshow(df_kappa, cmap='RdYlGn', vmin=-1, vmax=1) # Red (ruim) to Green (bom)
    fig.colorbar(cax)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(anotadores)
    ax.set_yticklabels(anotadores)
    
    # Adiciona os valores numéricos dentro dos quadrados do Heatmap
    for i in range(n):
        for j in range(n):
            val = df_kappa.iloc[i, j]
            if not np.isnan(val):
                cor_texto = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=cor_texto, fontsize=10)
                
    plt.title("Heatmap: Kappa de Cohen Pairwise", pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_RELATORIOS, "heatmap_pairwise.png"), dpi=300)
    plt.close()

    print(" └── Relatórios clássicos (Gráficos e Tabelas Base) restaurados e gerados com sucesso.")

"""
    Gera uma Matriz de Confusão Agregada comparando a classificação 
    dos anotadores humanos contra o consenso original da IA.
    Isso nos ajuda a identificar os padrões exatos de desvio 
    (ex: descobrir se a maioria dos humanos classifica GZ da IA como NT).
"""
def gerar_matriz_confusao_agregada(df_raw):
    print("\n--- Gerando Matriz de Confusão Agregada (IA vs. Humanos) ---")
    
    # 1. Recuperar o "Gabarito" da IA
    # Como a amostra foi extraída usando consenso absoluto prévio, 
    # podemos aplicar os limiares padrão do Detoxify no score bruto para 
    # reconstruir qual era a 'classificacao_ia' original daquela tira.
    t1_default = 0.20
    t2_default = 0.80
    
    def reconstruir_classe_ia(score):
        if score < t1_default: return LABEL_NT
        elif score >= t2_default: return LABEL_T
        else: return LABEL_GZ
        
    df_plot = df_raw.copy()
    df_plot['Classe_IA'] = df_plot[COLUNA_TOXICITY].apply(reconstruir_classe_ia)
    
    # 2. Calcular a Matriz de Confusão (Eixo Y = Verdade da IA, Eixo X = Predição Humana)
    labels_ordem = [LABEL_NT, LABEL_GZ, LABEL_T]
    cm = confusion_matrix(df_plot['Classe_IA'], df_plot[COLUNA_ROTULO], labels=labels_ordem)
    
    # 3. Plotagem em Alta Resolução com Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    # Configuração de Eixos
    ax.set_xticks(np.arange(len(labels_ordem)))
    ax.set_yticks(np.arange(len(labels_ordem)))
    ax.set_xticklabels(labels_ordem)
    ax.set_yticklabels(labels_ordem)
    
    # Posiciona os rótulos do eixo X na parte inferior (padrão de matriz de confusão)
    ax.xaxis.set_ticks_position('bottom')
    
    plt.xlabel('Predição (Anotadores Humanos)', weight='bold', labelpad=15)
    plt.ylabel('Verdadeiro (Consenso original da IA)', weight='bold', labelpad=15)
    plt.title('Matriz de Confusão Agregada: IA vs. Humanos', pad=20, weight='bold')
    
    # Adicionar os valores numéricos no centro de cada célula
    for i in range(len(labels_ordem)):
        for j in range(len(labels_ordem)):
            valor = cm[i, j]
            # Ajusta a cor do texto para contrastar com o fundo azul
            cor_texto = 'white' if valor > (cm.max() / 2) else 'black'
            ax.text(j, i, str(valor), ha='center', va='center', color=cor_texto, fontsize=12, weight='bold')
            
    plt.tight_layout()
    caminho_salvamento = os.path.join(DIRETORIO_RELATORIOS, "matriz_confusao_agregada.png")
    plt.savefig(caminho_salvamento, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" └── Gráfico salvo em: 'matriz_confusao_agregada.png'")

"""
    Gera um Gráfico de Dispersão (Scatter Plot) correlacionando a 
    confiança das IAs (scores de 0 a 1) com a concordância humana.
    Pontos verdes indicam que os humanos concordaram entre si, 
    pontos vermelhos indicam discordância.
"""
def gerar_grafico_dispersao_concordancia(df_raw, matriz_pivot):
    print("\n--- Gerando Gráfico de Dispersão (Confiança da IA vs. Discordância) ---")
    
    # 1. Determinar a concordância humana por SVT
    # Como a matriz_pivot tem as SVTs nas linhas e as anotações nas colunas,
    # contamos quantos rótulos únicos não-nulos cada SVT recebeu.
    concordancia_dict = {}
    for svt, row in matriz_pivot.iterrows():
        respostas_validas = row.dropna().unique()
        if len(respostas_validas) == 1:
            concordancia_dict[svt] = 'Concordaram'
        elif len(respostas_validas) > 1:
            concordancia_dict[svt] = 'Discordaram'
            
    # 2. Resgatar os scores originais da IA por SVT
    # Removemos as duplicatas baseando-se no ID da SVT para manter 1 score de IA por texto
    df_scores = df_raw.drop_duplicates(subset=[COLUNA_ID_SVT])[[COLUNA_ID_SVT, COLUNA_TOXICITY, COLUNA_PTOXICITY]].copy()
    
    # Mapeia a métrica de concordância para o dataframe de plotagem
    df_scores['Concordancia'] = df_scores[COLUNA_ID_SVT].map(concordancia_dict)
    df_plot = df_scores.dropna(subset=['Concordancia'])
    
    # 3. Plotagem em Alta Resolução
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Separa os grupos para marcações visuais distintas
    grupo_concordam = df_plot[df_plot['Concordancia'] == 'Concordaram']
    grupo_discordam = df_plot[df_plot['Concordancia'] == 'Discordaram']
    
    # Plot dos pontos de concordância (Bolinhas Verdes)
    ax.scatter(grupo_concordam[COLUNA_TOXICITY], grupo_concordam[COLUNA_PTOXICITY], 
               color='#2ECC71', alpha=0.6, edgecolors='black', label='Humanos Concordaram', s=50)
               
    # Plot dos pontos de discordância (X Vermelhos)
    ax.scatter(grupo_discordam[COLUNA_TOXICITY], grupo_discordam[COLUNA_PTOXICITY], 
               color='#E74C3C', alpha=0.7, edgecolors='black', label='Humanos Discordaram', s=60, marker='X')
               
    # Adiciona linhas de referência dos Thresholds originais da sua metodologia
    # Detoxify (Vertical): 0.20 (Max NT) e 0.80 (Min T)
    ax.axvline(x=0.20, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.80, color='gray', linestyle='--', alpha=0.5)
    # Perspective API (Horizontal): 0.20 (Max NT) e 0.40 (Min T)
    ax.axhline(y=0.20, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.40, color='gray', linestyle='--', alpha=0.5)
    
    # Configuração de Eixos
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Score de Confiança: Detoxify (0 a 1)', weight='bold')
    ax.set_ylabel('Score de Confiança: Perspective API (0 a 1)', weight='bold')
    ax.set_title('Gráfico de Dispersão: Discordância Humana vs. Confiança da IA', pad=20, weight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    caminho_salvamento = os.path.join(DIRETORIO_RELATORIOS, "dispersao_concordancia_ia.png")
    plt.savefig(caminho_salvamento, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" └── Gráfico de dispersão salvo em: 'dispersao_concordancia_ia.png'")

"""
    Cruza os dados da matriz de anotação com os textos originais ('tiras') 
    para classificar e exportar as SVTs que causaram falhas severas de concordância.
    Permite a análise qualitativa (Error Analysis) para revisão do Codebook.
"""
def gerar_relatorio_qualitativo_erros(df_raw, matriz_pivot):
    print("\n--- Gerando Relatório Qualitativo de Divergências ---")
    
    # 1. Isolar os textos e scores da IA (Mantendo 1 registro por SVT)
    df_base = df_raw.drop_duplicates(subset=[COLUNA_ID_SVT])[[COLUNA_ID_SVT, COLUNA_TEXTO, COLUNA_TOXICITY, COLUNA_PTOXICITY]].copy()
    
    # 2. Mesclar as anotações humanas lado a lado com os textos
    df_analise = df_base.merge(matriz_pivot, left_on=COLUNA_ID_SVT, right_index=True)
    
    anotadores = matriz_pivot.columns.tolist()
    classificacoes = []
    
    # 3. Classificar o tipo de divergência cognitiva
    for idx, row in df_analise.iterrows():
        # Extrai apenas as respostas humanas não-nulas para esta SVT
        respostas_humanas = row[anotadores].dropna().tolist()
        
        if len(respostas_humanas) == 2:
            voto1, voto2 = respostas_humanas[0], respostas_humanas[1]
            set_votos = {voto1, voto2}
            
            score_ia_detoxify = row[COLUNA_TOXICITY]
            
            # Regra 1: Discordância Crítica (Opiniões Opostas - Colapso do Construto)
            if set_votos == {LABEL_NT, LABEL_T}:
                classificacoes.append("1. Discordância Crítica (NT vs T)")
                
            # Regra 2: Ruído Adjacente (Uso da Zona Cinzenta como refúgio)
            elif LABEL_GZ in set_votos and len(set_votos) > 1:
                classificacoes.append("2. Ruído Adjacente (Insegurança GZ)")
                
            # Regra 3: Triplo Falso (Certeza Absoluta da IA Tóxica vs Consenso Humano NT)
            elif len(set_votos) == 1 and voto1 == LABEL_NT and score_ia_detoxify >= 0.80:
                classificacoes.append("3. Falso Positivo Extremo da IA (Humanos NT vs IA >0.8)")
                
            else:
                classificacoes.append("Consenso Regular / Outros")
        else:
            classificacoes.append("Dados Incompletos")
            
    df_analise['Tipo_Divergencia'] = classificacoes
    
    # 4. Filtrar apenas os casos de interesse e limpar os que deram certo
    df_erros = df_analise[df_analise['Tipo_Divergencia'] != "Consenso Regular / Outros"]
    df_erros = df_erros[df_erros['Tipo_Divergencia'] != "Dados Incompletos"]
    
    # Ordenação alfabética faz a Crítica aparecer no topo (graças à numeração que adicionamos)
    df_erros = df_erros.sort_values(by='Tipo_Divergencia')
    
    caminho_salvamento = os.path.join(DIRETORIO_RELATORIOS, "relatorio_qualitativo_divergencias.csv")
    df_erros.to_csv(caminho_salvamento, index=False, encoding='utf-8-sig') # utf-8-sig resolve bugs de acentuação no Excel
    
    print(f" └── Encontradas {len(df_erros)} SVTs com conflito semântico severo.")
    print(f" └── Relatório textutal (Error Analysis) salvo em: 'relatorio_qualitativo_divergencias.csv'")

# =====================================================================
# EXECUÇÃO DO PIPELINE
# =====================================================================
if __name__ == "__main__":
    try:
        # 1. Extração e Tratamento
        df_dados_brutos = carregar_e_preparar_dados(DIRETORIO_DADOS)
        matriz_esparsa = criar_matriz_esparsa(df_dados_brutos)
        
        # 2. Métricas Globais Acadêmicas
        reportar_metricas(df_dados_brutos, matriz_esparsa)
        
        # 3. Geração de Relatórios Comportamentais Base
        gerar_relatorios_estatisticos_visuais(df_dados_brutos, matriz_esparsa)
        
        # 4. Otimização de Limiares (Threshold Tuning)
        otimizar_thresholds_por_anotador(df_dados_brutos)
        
        # 5. Diagnóstico de Erros (Nova Matriz de Confusão)
        gerar_matriz_confusao_agregada(df_dados_brutos)
        
        # 6. Dispersão de Confiança vs. Discordância
        gerar_grafico_dispersao_concordancia(df_dados_brutos, matriz_esparsa)
        
        # 7. Relatório Textual para Análise Qualitativa
        gerar_relatorio_qualitativo_erros(df_dados_brutos, matriz_esparsa)
        
    except Exception as erro:
        print(f"\n[FALHA NA EXECUÇÃO] {erro}")