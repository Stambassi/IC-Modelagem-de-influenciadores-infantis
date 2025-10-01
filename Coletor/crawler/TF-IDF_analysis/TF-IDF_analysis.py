import pandas as pd
import re
import nltk
from pathlib import Path
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

console = Console()

nlp = spacy.load('pt_core_news_lg', disable=['parser', 'ner'])

nlp.max_length = 1600000 

'''
    Baixar a lista de stop words em Português do NTLK
'''
try:
    nltk.data.find('corpora/stopwords')
except Exception as e:
    console.print("[yellow]Baixando a lista de stopwords do NLTK para português...[/yellow]")
    nltk.download('stopwords')

'''
    Função para concatenar todas as tiras de um youtuber em uma string única
    @param youtuber_name - Nome do youtuber a ser analisado
    @param base_folder - Nome da pasta raiz (por padrão, é 'files')
    @return str - String única com todas as tiras de um youtuber concatenadas
'''
def agregar_texto_youtuber(youtuber_name: str, base_folder: str = 'files') -> str:
    #console.print(f"Agregando textos para [cyan]{youtuber_name}[/cyan]...")

    youtuber_path = Path(base_folder) / youtuber_name
    all_text = []

    if not youtuber_path.is_dir():
        console.print(f"[red]Diretório não encontrado para {youtuber_name}[/red]")
        return ""

    for csv_path in youtuber_path.rglob('tiras_video.csv'):
        try:
            # Ler arquivo de tiras
            df = pd.read_csv(csv_path)

            # .dropna() remove qualquer valor NaN que possa existir na coluna
            all_text.append(' '.join(df['tiras'].dropna().astype(str)))

        except Exception as e:
            console.print(f"  [yellow]Aviso: Não foi possível ler o arquivo {csv_path}. Erro: {e}[/yellow]")
    
    return ' '.join(all_text)

'''
    Função para limpar o texto: minúsculas, remove pontuação/números e stopwords
    @param text - Texto a ser limpo
    @param customize_stopwords - Flag para testar se devem ser consideradas stopwords customizadas
    @return str - Texto limpo como uma única string
'''
def preprocessar_texto_nltk(text: str, customize_stopwords: bool = False) -> str:
    # Tornar todas os caracteres minúsculos
    text = text.lower()

    # Remover tudo que não for letra ou espaço
    text = re.sub(r'[^a-z\s]', '', text, re.UNICODE)
    
    # Tokenizar o texto
    tokens = text.split()
    
    # Remover stopwords
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    
    # Adicionar stopwords personalizadas (palavras comuns no contexto de YouTube)
    #custom_stopwords = ['jogo', 'game', 'canal', 'vídeo', 'pessoal', 'galera', 'roblox', 'tex', 'hs']
    #stopwords_pt.extend(custom_stopwords)
    
    tokens_limpos = [word for word in tokens if word not in stopwords_pt and len(word) > 2]
    
    return ' '.join(tokens_limpos)

'''
    Função para limpar o texto: minúsculas, remove pontuação/números estopwords; e filtrar por adjetivos e substantivos
    @param text - Texto a ser limpo
    @return str - Texto limpo como uma única string
'''
def preprocessar_texto_spacy(text: str) -> str:
    # Tornar todas os caracteres minúsculos
    text = text.lower()
    
    # Processar o texto com o spaCy
    doc = nlp(text)
    
    # Lista das classes gramaticais que devem ser mantidas
    allowed_pos = ['NOUN', 'PROPN', 'ADJ'] # Substantivos, Nomes Próprios, Adjetivos
    
    # Lista de stopwords padrão do spaCy (mais completa)
    stopwords_pt = spacy.lang.pt.stop_words.STOP_WORDS
    
    # Adicionar stopwords personalizadas
    custom_stopwords = {
        'tex', 'hs', 'ju', 'julia', 'minegirl', 'roblox', # Palavras base
        'gente', 'pessoal', 'galera', 'cara', 'amigo', 'mames', 'cris', # Ruído social/Nomes
        'jogo', 'casa', 'coisa' # Domínio comum
    } 
    
    tokens_limpos = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in allowed_pos and token.lemma_.lower() not in stopwords_pt and token.lemma_.lower() not in custom_stopwords and len(token.lemma_) > 2
    ]
    
    return ' '.join(tokens_limpos)

'''
    Função para aplicar o TF-IDF no texto agregado de cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param preprocessar - Função de pré-processamento do texto
    @param n_top_words - Número de palavras do ranking a serem persistidas
'''
def analisar_frequencia(youtubers_list: list[str], preprocessar, n_top_words: int = 20) -> None:
    # Criar o "corpus": uma lista onde cada item é o texto completo e agregado de um youtuber
    corpus_bruto = [agregar_texto_youtuber(yt) for yt in lista_youtubers]
    
    # Pré-processar cada documento do corpus
    console.print("\nIniciando pré-processamento de todos os textos...")
    corpus_limpo = [preprocessar(doc) for doc in corpus_bruto]
    
    # Calcular o TF-IDF
    console.print("\nCalculando as pontuações TF-IDF...")
    
    # Inicializa o vetorizador
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2)) # Limita para as 1000 palavras mais relevantes no geral

    # Aplica o TF-IDF ao corpus limpo
    tfidf_matrix = vectorizer.fit_transform(corpus_limpo)
    
    # Identificar os nomes das palavras (features) que o vetorizador aprendeu
    feature_names = vectorizer.get_feature_names_out()
    
    # Extrair, apresentar e persistir os resultados
    console.print("\n--- [bold green]Extraindo e Salvando Rankings Individuais por YouTuber (TF-IDF)[/bold green] ---")

    # Iterar sobre cada youtuber para processar e salvar seu respectivo arquivo
    for i, youtuber in enumerate(lista_youtubers):        
        # Lista para armazenar os resultados apenas desse youtuber
        resultados_youtuber = []
        
        # Identificar a linha da matriz correspondente a este youtuber
        youtuber_tfidf_scores = tfidf_matrix[i].toarray().flatten()
        
        # Identificar os índices das N maiores pontuações
        top_indices = youtuber_tfidf_scores.argsort()[-n_top_words:][::-1]
        
        # Adiciona os resultados detalhados à lista desse youtuber
        for rank, idx in enumerate(top_indices):
            palavra = feature_names[idx]
            pontuacao = youtuber_tfidf_scores[idx]
            resultados_youtuber.append({
                'rank': rank + 1,
                'palavra_chave': palavra,
                'pontuacao_tfidf': pontuacao
            })

        # Converter a lista de resultados desse youtuber em um DataFrame
        df_youtuber = pd.DataFrame(resultados_youtuber)

        # Definir o nome da pasta específica para essa análise
        pasta_analise = 'frequencia_palavras'
        
        # Montar o caminho completo da pasta de saída
        caminho_saida_pasta = Path('files') / youtuber / pasta_analise
        caminho_saida_pasta.mkdir(parents=True, exist_ok=True)
        
        # Definir o nome do arquivo CSV de saída
        caminho_saida_csv = caminho_saida_pasta / 'ranking_tfidf.csv'
        
        # Salvar o DataFrame específico desse youtuber no arquivo
        df_youtuber.to_csv(caminho_saida_csv, index=False, encoding='utf-8')
        
        console.print(f"Ranking para [bold cyan]{youtuber}[/bold cyan] salvo em: [green]{caminho_saida_csv}[/green]")

if __name__ == '__main__':
    #lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']
    lista_youtubers = ['Julia MineGirl', 'Tex HS']

    #analisar_frequencia(lista_youtubers, preprocessar_texto_nltk)
    analisar_frequencia(lista_youtubers, preprocessar_texto_spacy, 50)