import pandas as pd
from pathlib import Path
from rich.console import Console
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

# Configurações de Caminho
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent
OUTPUT_FOLDER = PROJECT_ROOT / "tabelas"
BASE_DIR = Path(PROJECT_ROOT.parent.parent / 'files')

# Criar pasta de tabelas se não existir
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

'''
    Função para aplicar a regra de discretização definida na metodologia

    @param score - Valor de toxicidade a ser discretizado
    @returns NT, GZ ou T
'''
def discretizar_toxicidade(score):
    if score < 0.20:
        return "NT"
    elif score < 0.80:
        return "GZ"
    else:
        return "T"

'''
    Função que varre os arquivos tiras_video.csv e calcula métricas comparativas 
    entre Detoxify e Perspective API, permitindo thresholds customizados.

    @param d_limiar_inf - Threshold inferior do Detoxify (NT -> GZ)
    @param d_limiar_sup - Threshold superior do Detoxify (GZ -> T)
    @param p_limiar_inf - Threshold inferior do Perspective (NT -> GZ)
    @param p_limiar_sup - Threshold superior do Perspective (GZ -> T)
'''
def gerar_estatisticas_iniciais(d_limiar_inf=0.20, d_limiar_sup=0.80, p_limiar_inf=0.20, p_limiar_sup=0.40):
    console.print(f"[bold cyan]Iniciando levantamento estatístico (Thresholds: D[{d_limiar_inf}/{d_limiar_sup}] P[{p_limiar_inf}/{p_limiar_sup}])...[/bold cyan]")
    
    dados_acumulados = []
    videos_processados = 0
    
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            col_detox = 'toxicity' 
            col_persp = 'p_toxicity'
            
            if col_detox in df.columns and col_persp in df.columns:
                videos_processados += 1
                dados_acumulados.append(df[[col_detox, col_persp]])
        except Exception as e:
            console.print(f"[red]Erro ao ler {csv_path}: {e}[/red]")

    if not dados_acumulados:
        console.print("[yellow]Nenhum dado processado com ambas as ferramentas foi encontrado.[/yellow]")
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True)

    # Discretização customizada baseada nos parâmetros da função
    def discretizar_custom(score, inf, sup):
        if score < inf: return "NT"
        elif score < sup: return "GZ"
        else: return "T"

    df_total['cat_detox'] = df_total['toxicity'].apply(lambda x: discretizar_custom(x, d_limiar_inf, d_limiar_sup))
    df_total['cat_persp'] = df_total['p_toxicity'].apply(lambda x: discretizar_custom(x, p_limiar_inf, p_limiar_sup))

    stats = []

    for ferramenta, col in [("Detoxify", "toxicity"), ("Perspective", "p_toxicity")]:
        cat_col = 'cat_detox' if ferramenta == "Detoxify" else 'cat_persp'
        proporcoes = df_total[cat_col].value_counts(normalize=True).to_dict()
        quantidades = df_total[cat_col].value_counts(normalize=False).to_dict()
        
        stats.append({
            "Ferramenta": ferramenta,
            "Média": round(df_total[col].mean(), 4),
            "Mediana": round(df_total[col].median(), 4),
            "Desvio Padrão": round(df_total[col].std(), 4),
            "Qtd NT": quantidades.get("NT", 0),
            "Qtd GZ": quantidades.get("GZ", 0),
            "Qtd T": quantidades.get("T", 0),
            "Prop. NT": f"{proporcoes.get('NT', 0):.2%}",
            "Prop. GZ": f"{proporcoes.get('GZ', 0):.2%}",
            "Prop. T": f"{proporcoes.get('T', 0):.2%}"
        })

    df_stats = pd.DataFrame(stats)
    
    # Salvar CSV
    csv_path = OUTPUT_FOLDER / "estatisticas_iniciais_comparacao.csv"
    df_stats.to_csv(csv_path, index=False)
    
    # Gerar Imagem
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    tabela_visual = ax.table(
        cellText=df_stats.values, 
        colLabels=df_stats.columns, 
        cellLoc='center', 
        loc='center',
        colColours=["#d1e7dd"] * len(df_stats.columns)
    )
    tabela_visual.auto_set_font_size(False)
    tabela_visual.set_fontsize(10)
    tabela_visual.scale(1.2, 1.8)
    
    plt.title(f"Estatísticas Iniciais: Detoxify vs Perspective\n(Limiares Customizados)", fontsize=14, pad=25)
    
    image_path = OUTPUT_FOLDER / "estatisticas_iniciais_comparacao.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    console.print(f"[bold green]Arquivos gerados para {videos_processados} vídeos:[/bold green]\n - {csv_path.name}\n - {image_path.name}")

'''
    Função para calcular Pearson (linear), MAE (erro absoluto) e Cohen's Kappa (categorias)
    Salva o resultado em tabelas/metricas_concordancia.csv
'''
def gerar_metricas_concordancia():
    console.print("[bold cyan]Calculando métricas de concordância técnica...[/bold cyan]")
    
    dados_acumulados = []
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            if 'toxicity' in df.columns and 'p_toxicity' in df.columns:
                dados_acumulados.append(df[['toxicity', 'p_toxicity']])
        except Exception as e:
            continue

    if not dados_acumulados:
        console.print("[red]Dados insuficientes para calcular métricas.[/red]")
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True)

    # Limpeza de NaNs
    df_total = df_total.dropna(subset=['toxicity', 'p_toxicity'])
    
    if df_total.empty:
        console.print("[red]Erro: Sem dados válidos após limpeza.[/red]")
        return

    # 1. Preparar labels discretos para o Kappa
    y_detox = df_total['toxicity'].apply(discretizar_toxicidade)
    y_persp = df_total['p_toxicity'].apply(discretizar_toxicidade)

    # 2. Cálculo das Métricas
    corr_p, _ = pearsonr(df_total['toxicity'], df_total['p_toxicity'])
    mae = mean_absolute_error(df_total['toxicity'], df_total['p_toxicity'])
    kappa = cohen_kappa_score(y_detox, y_persp)

    # 3. Persistência em CSV
    metricas = {
        "Métrica": ["Pearson (r)", "MAE", "Cohen's Kappa"],
        "Valor": [round(corr_p, 4), round(mae, 4), round(kappa, 4)],
        "Interpretação": [
            "Força da relação linear",
            "Divergência média de score",
            "Concordância além do acaso"
        ]
    }
    
    df_metricas = pd.DataFrame(metricas)
    csv_path = OUTPUT_FOLDER / "metricas_concordancia.csv"
    df_metricas.to_csv(csv_path, index=False)

    # 4. Gerar Imagem da Tabela (PNG)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Criando a tabela visual com cores para destacar os títulos
    tabela_visual = ax.table(
        cellText=df_metricas.values, 
        colLabels=df_metricas.columns, 
        cellLoc='center', 
        loc='center',
        colColours=["#d1e7dd"] * len(df_metricas.columns) # Verde claro
    )
    tabela_visual.auto_set_font_size(False)
    tabela_visual.set_fontsize(11)
    tabela_visual.scale(1.2, 2)
    
    plt.title("Métricas de Similaridade e Concordância", fontsize=14, pad=10)
    
    image_path = OUTPUT_FOLDER / "metricas_concordancia.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    console.print(f"[green]Arquivos de concordância gerados:[/green]\n - {csv_path.name}\n - {image_path.name}")

'''
    Função para gerar uma Matriz de Confusão comparando as categorias (NT, GZ, T) 
    entre Detoxify e Perspective. Salva em tabelas/matriz_confusao.csv
'''
def gerar_matriz_confusao():
    console.print("[bold cyan]Gerando Matriz de Confusão e Heatmap (Detoxify vs Perspective)...[/bold cyan]")
    
    dados_acumulados = []
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            if 'toxicity' in df.columns and 'p_toxicity' in df.columns:
                dados_acumulados.append(df[['toxicity', 'p_toxicity']])
        except Exception:
            continue

    if not dados_acumulados:
        console.print("[red]Dados insuficientes para gerar a matriz.[/red]")
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True).dropna()

    # Criar as labels discretas
    df_total['Detoxify_Label'] = df_total['toxicity'].apply(discretizar_toxicidade)
    df_total['Perspective_Label'] = df_total['p_toxicity'].apply(discretizar_toxicidade)

    # Gerar a matriz
    ordem = ["NT", "GZ", "T"]
    matriz = pd.crosstab(
        df_total['Detoxify_Label'], 
        df_total['Perspective_Label'], 
        dropna=False
    )

    # Garantir integridade da matriz e reordenar
    for cat in ordem:
        if cat not in matriz.index: matriz.loc[cat] = 0
        if cat not in matriz.columns: matriz[cat] = 0
    matriz = matriz.reindex(index=ordem, columns=ordem).fillna(0).astype(int)

    # 1. Salvar CSV
    csv_path = OUTPUT_FOLDER / "matriz_confusao.csv"
    matriz.to_csv(csv_path)

    # 2. Gerar Heatmap (PNG)
    plt.figure(figsize=(10, 8))
    
    # cmap='YlGnBu' cria um degradê do amarelo ao azul escuro
    # annot=True coloca os números dentro das células
    # fmt='d' garante que os números sejam inteiros
    sns.heatmap(matriz, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                xticklabels=ordem, yticklabels=ordem)
    
    plt.title("Matriz de Confusão: Detoxify vs Perspective API", fontsize=15, pad=20)
    plt.xlabel("Perspective API (Labels)", fontsize=12)
    plt.ylabel("Detoxify (Labels)", fontsize=12)
    
    image_path = OUTPUT_FOLDER / "matriz_confusao.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Cálculo de acurácia para o console
    acuracia = (np.diag(matriz).sum() / matriz.values.sum()) * 100
    console.print(f"[green]Arquivos gerados:[/green]\n - {csv_path.name}\n - {image_path.name}")
    console.print(f"[yellow]Acurácia de concordância: {acuracia:.2f}%[/yellow]")

'''
    Função para calcular a concordância relativa entre os modelos variando o threshold superior (GZ -> T).
    Calcula o quanto um modelo concorda com o outro baseado no total de labels 'T' de cada um.
'''
def analisar_sensibilidade_threshold():
    console.print("[bold cyan]Iniciando análise de sensibilidade de thresholds (Concordância Relativa)...[/bold cyan]")
    
    dados_acumulados = []
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            if 'toxicity' in df.columns and 'p_toxicity' in df.columns:
                dados_acumulados.append(df[['toxicity', 'p_toxicity']])
        except:
            continue

    if not dados_acumulados:
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True).dropna()
    
    thresholds_superiores = [0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_inferior = 0.20
    resultados = []

    for t_sup in thresholds_superiores:
        def discretizar_variavel(score):
            if score < threshold_inferior: return "NT"
            elif score < t_sup: return "GZ"
            else: return "T"

        labels_detox = df_total['toxicity'].apply(discretizar_variavel)
        labels_persp = df_total['p_toxicity'].apply(discretizar_variavel)

        sao_iguais = (labels_detox == labels_persp)

        total_t_detox = (labels_detox == "T").sum()
        concordancia_t_detox = ((labels_detox == "T") & (labels_persp == "T")).sum()
        prop_visto_pelo_detox = concordancia_t_detox / total_t_detox if total_t_detox > 0 else 0

        total_t_persp = (labels_persp == "T").sum()
        concordancia_t_persp = ((labels_persp == "T") & (labels_detox == "T")).sum()
        prop_visto_pela_persp = concordancia_t_persp / total_t_persp if total_t_persp > 0 else 0

        prop_geral = sao_iguais.mean()

        resultados.append({
            "Threshold Superior": t_sup,
            "Conf. Detox (T)": f"{prop_visto_pelo_detox:.2%}",
            "Conf. Persp (T)": f"{prop_visto_pela_persp:.2%}",
            "Acurácia Geral": f"{prop_geral:.2%}",
            "Qtd T Detox": total_t_detox,
            "Qtd T Persp": total_t_persp
        })

    # 1. Salvar CSV
    df_sensibilidade = pd.DataFrame(resultados)
    csv_path = OUTPUT_FOLDER / "analise_sensibilidade_concordancia.csv"
    df_sensibilidade.to_csv(csv_path, index=False)

    # 2. Gerar Tabela em Imagem (PNG)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    tabela_visual = ax.table(
        cellText=df_sensibilidade.values, 
        colLabels=df_sensibilidade.columns, 
        cellLoc='center', 
        loc='center',
        colColours=["#d1e7dd"] * len(df_sensibilidade.columns) # Verde claro
    )
    
    tabela_visual.auto_set_font_size(False)
    tabela_visual.set_fontsize(10)
    tabela_visual.scale(1.2, 2.0)
    
    plt.title(f"Análise de Sensibilidade (Threshold Inferior Fixo: {threshold_inferior})", 
        fontsize=14, pad=20)
    
    image_path = OUTPUT_FOLDER / "analise_sensibilidade_concordancia.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    console.print(f"[green]Arquivos de sensibilidade gerados:[/green]\n - {csv_path.name}\n - {image_path.name}")

'''
    Função para buscar o threshold superior (GZ -> T) que atinge um nível de concordância X
    Mantém o threshold inferior fixo em 0.20
    
    @param nivel_alvo: Float entre 0.0 e 1.0 (ex: 0.70 para 70%)
'''
def encontrar_threshold_por_concordancia(nivel_alvo: float):
    console.print(f"\n[bold cyan]Buscando threshold para concordância de {nivel_alvo*100:.1f}%...[/bold cyan]")
    
    dados_acumulados = []
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            if 'toxicity' in df.columns and 'p_toxicity' in df.columns:
                dados_acumulados.append(df[['toxicity', 'p_toxicity']])
        except:
            continue

    if not dados_acumulados:
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True).dropna()
    threshold_inferior = 0.20
    
    thresholds_testados = np.linspace(0.40, 0.99, 60)
    hist_conf_detox = [] 
    hist_conf_persp = [] 
    
    melhor_t_encontrado = None
    maior_concordancia_global = 0.0

    for t_sup in thresholds_testados:
        def discretizar_teste(score):
            if score < threshold_inferior: return "NT"
            elif score < t_sup: return "GZ"
            else: return "T"

        labels_detox = df_total['toxicity'].apply(discretizar_teste)
        labels_persp = df_total['p_toxicity'].apply(discretizar_teste)

        concordancia_global = (labels_detox == labels_persp).mean()
        
        mask_t_detox = (labels_detox == "T")
        mask_t_persp = (labels_persp == "T")
        
        # Usar np.nan se o denominador for zero para evitar queda falsa no gráfico
        conf_detox = (mask_t_detox & (labels_persp == "T")).sum() / mask_t_detox.sum() if mask_t_detox.sum() > 0 else np.nan
        conf_persp = (mask_t_persp & (labels_detox == "T")).sum() / mask_t_persp.sum() if mask_t_persp.sum() > 0 else np.nan
        
        hist_conf_detox.append(conf_detox)
        hist_conf_persp.append(conf_persp)
        
        if melhor_t_encontrado is None and concordancia_global >= nivel_alvo:
            melhor_t_encontrado = t_sup
        
        if concordancia_global > maior_concordancia_global:
            maior_concordancia_global = concordancia_global
            melhor_t_temp = t_sup

    t_final = melhor_t_encontrado if melhor_t_encontrado else melhor_t_temp

    # 1. Salvar CSV
    resultado = {
        "Nivel_Alvo_Desejado": [nivel_alvo],
        "Threshold_Encontrado": [round(t_final, 3)],
        "Concordancia_Global_Obtida": [round(maior_concordancia_global, 4)],
        "Status": ["Sucesso" if melhor_t_encontrado else "Alvo não atingido"]
    }
    pd.DataFrame(resultado).to_csv(OUTPUT_FOLDER / "busca_threshold_concordancia.csv", index=False)

    # 2. Gerar Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
    
    # Plot 1
    ax1.plot(thresholds_testados, hist_conf_detox, color='#1f77b4', linewidth=2, label='Concordância Perspective')
    ax1.axhline(y=nivel_alvo, color='#e74c3c', linestyle='--', alpha=0.6, label=f'Alvo ({nivel_alvo*100}%)')
    ax1.set_title("Perspectiva Detoxify (Concordância da Perspective em T)", fontsize=11)
    ax1.set_ylabel("Proporção de Concordância")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Plot 2
    ax2.plot(thresholds_testados, hist_conf_persp, color='#2ca02c', linewidth=2, label='Concordância Detoxify')
    ax2.axhline(y=nivel_alvo, color='#e74c3c', linestyle='--', alpha=0.6, label=f'Alvo ({nivel_alvo*100}%)')
    ax2.set_title("Perspectiva Perspective (Concordância do Detoxify em T)", fontsize=11)
    ax2.set_xlabel("Threshold Superior Testado (GZ -> T)")
    ax2.set_ylabel("Proporção de Concordância")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.suptitle(f"Análise de Sensibilidade de Concordância por Modelo\n(Interrupção de linha indica ausência de amostras 'T')", fontsize=13, y=0.96)
    
    plt.savefig(OUTPUT_FOLDER / "busca_threshold_concordancia.png", bbox_inches='tight', dpi=300)
    plt.close()

    console.print(f"[green]Gráficos gerados com tratamento de NaNs para evitar quedas artificiais.[/green]") 

'''
    Função para isolar e persistir todas as tiras que atingiram o limiar crítico
    na Perspective API (score > 0.80), incluindo os dados do Detoxify para comparação.
'''
def persistir_casos_criticos_perspective(threshold_alvo: float = 0.70):
    console.print(f"[bold magenta]Isolando casos críticos da Perspective (Score > {threshold_alvo})...[/bold magenta]")
    
    amostras_criticas = []
    
    # Busca recursiva em todas as pastas de vídeos
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            
            # Verificar se as colunas necessárias existem
            if 'p_toxicity' in df.columns and 'toxicity' in df.columns:
                # Filtrar apenas onde a Perspective foi agressiva
                df_filtrado = df[df['p_toxicity'] > threshold_alvo].copy()
                
                if not df_filtrado.empty:
                    # Adicionar metadados para saber de qual vídeo/youtuber veio a tira
                    # (Extraindo o nome do youtuber e ID do vídeo do caminho do arquivo)
                    partes_caminho = csv_path.parts
                    # Ajuste os índices [-3] e [-2] conforme a estrutura (ex: files/Youtuber/Ano/Mês/Video/tiras_video.csv)
                    df_filtrado['youtuber'] = partes_caminho[-5]
                    df_filtrado['video_id'] = partes_caminho[-3]
                    
                    amostras_criticas.append(df_filtrado)
        except Exception as e:
            continue

    if amostras_criticas:
        df_final = pd.concat(amostras_criticas, ignore_index=True)
        
        # Ordenar pelos casos mais tóxicos segundo a Perspective
        df_final = df_final.sort_values(by='p_toxicity', ascending=False)
        
        output_path = OUTPUT_FOLDER / "casos_criticos_perspective.csv"
        df_final.to_csv(output_path, index=False)
        
        console.print(f"[green]Sucesso! {len(df_final)} casos críticos exportados para:[/green] {output_path}")
        
        # Exibir os top 5 casos no console para inspeção rápida
        console.print("\n[bold yellow]Amostra dos Casos Identificados:[/bold yellow]")
        print(df_final[['youtuber', 'p_toxicity', 'toxicity', 'tiras']].head().to_string(index=False))
    else:
        console.print("[yellow]Nenhuma tira encontrada com score da Perspective acima do limiar informado.[/yellow]")

'''
    Função para calcular a intersecção de labels entre os modelos variando o threshold
    da Perspective API para encontrar um critério de combinação (Ensemble) otimizado.
'''
def gerar_tabela_interseccao_modelos():
    console.print("[bold cyan]Iniciando análise de intersecção com múltiplos thresholds para Perspective...[/bold cyan]")
    
    dados_acumulados = []
    for csv_path in BASE_DIR.rglob("tiras_video.csv"):
        try:
            df = pd.read_csv(csv_path)
            if 'toxicity' in df.columns and 'p_toxicity' in df.columns:
                dados_acumulados.append(df[['toxicity', 'p_toxicity']])
        except:
            continue

    if not dados_acumulados:
        console.print("[red]Nenhum dado encontrado.[/red]")
        return

    df_total = pd.concat(dados_acumulados, ignore_index=True).dropna()
    total = len(df_total)

    # Thresholds do Perspective para testar o Consenso com Detoxify (> 0.80)
    thresholds_persp = [0.20, 0.30, .40, 0.50, 0.60, 0.70, 0.80]
    cenarios = []

    # 1. Testar variações de Consenso (Ambos identificam como potencial Toxicidade)
    for t_persp in thresholds_persp:
        qtd = len(df_total[(df_total['toxicity'] > 0.80) & (df_total['p_toxicity'] > t_persp)])
        cenarios.append({
            "Cenário": f"Consenso entre Detox e Persp",
            "Regra": f"Detox > 0.80 E Persp > {t_persp}",
            "Quantidade": qtd,
            "Tipo": "Consenso"
        })

    # 2. Testar Filtro Conservador
    cenarios.append({
        "Cenário": "Filtro Conservador Base",
        "Regra": "Detox > 0.80 E Persp > 0.20",
        "Quantidade": len(df_total[(df_total['toxicity'] > 0.80) & (df_total['p_toxicity'] > 0.20)]),
        "Tipo": "Filtro"
    })

    # 3. Testar Divergências
    for t_persp in thresholds_persp:
        qtd = len(df_total[(df_total['toxicity'] > 0.80) & (df_total['p_toxicity'] < t_persp)])
        cenarios.append({
            "Cenário": f"Divergência (Persp ignora T)",
            "Regra": f"Detox > 0.80 E Persp < {t_persp}",
            "Quantidade": qtd,
            "Tipo": "Divergência"
        })

    df_cenarios = pd.DataFrame(cenarios)
    df_cenarios['% do Total'] = (df_cenarios['Quantidade'] / total).map('{:.2%}'.format)

    # 1. Salvar CSV
    csv_path = OUTPUT_FOLDER / "interseccao_modelos_variacao.csv"
    df_cenarios.to_csv(csv_path, index=False)

    # 2. Gerar Tabela Visual
    # Aumentando o tamanho para comportar mais linhas
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    tabela = ax.table(
        cellText=df_cenarios.values,
        colLabels=df_cenarios.columns,
        cellLoc='center',
        loc='center',
        colColours=["#d1e7dd"] * len(df_cenarios.columns)
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.0, 2.0)
    
    plt.title("Estratégias de Ensemble: Variação de Threshold Perspective (Detoxify fixo em 0.80)", fontsize=14, pad=20)
    
    image_path = OUTPUT_FOLDER / "interseccao_modelos_variacao.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    console.print(f"[green]Análise de intersecção com variações concluída![/green]\n - {csv_path.name}\n - {image_path.name}")

if __name__ == "__main__":
    gerar_estatisticas_iniciais()

    gerar_metricas_concordancia()

    gerar_matriz_confusao()

    analisar_sensibilidade_threshold()

    encontrar_threshold_por_concordancia(0.9)

    persistir_casos_criticos_perspective(0.50)

    gerar_tabela_interseccao_modelos()