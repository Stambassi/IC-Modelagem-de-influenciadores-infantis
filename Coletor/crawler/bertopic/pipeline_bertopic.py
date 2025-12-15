import pandas as pd
from rich.console import Console
from InquirerPy import prompt

# Importações dos seus módulos modulares
from visualizar_topicos import visualizar_bertopic
from otimizar_bertopic import otimizar_BERTopic, salvar_BERTopic
from preparar_dados import get_dados

console = Console()

# Mapa de categorias
MAPA_YOUTUBERS_CATEGORIA = {
    'Amy Scarlet': 'Roblox',
    'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft',
    'Julia MineGirl': 'Roblox',
    'Kass e KR': 'Minecraft',
    'Lokis': 'Roblox',
    'Luluca Games': 'Roblox',
    'Papile': 'Roblox',
    'Robin Hood Gamer': 'Minecraft',
    'TazerCraft': 'Minecraft',
    'Tex HS': 'Misto'
}

'''
    Retorna a lista de youtubers com base no filtro solicitado
    
    @param nome_grupo - Nome do grupo ('Geral', categoria ou nome do youtuber)
    @return lista - Lista de strings com nomes dos youtubers
'''
def obter_lista_youtubers(nome_grupo):
    if nome_grupo == "Geral":
        return list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    # Verifica se é uma categoria
    categorias = set(MAPA_YOUTUBERS_CATEGORIA.values())
    if nome_grupo in categorias:
        return [y for y, cat in MAPA_YOUTUBERS_CATEGORIA.items() if cat == nome_grupo]
    
    # Verifica se é um youtuber individual
    if nome_grupo in MAPA_YOUTUBERS_CATEGORIA:
        return [nome_grupo]
        
    console.print(f"[bold red]Aviso:[/] Grupo '{nome_grupo}' não encontrado. Retornando vazio.")
    return []

# --- Stop words ---
custom_sWords = {
    # Básicas e Gírias
    "aqui", "pra", "velho", "né", "tá", "mano", "ah", "beleza", 
    "olá", "tô", "gente", "ta", "olha", "pá", "vi", "ai", "será",
    "pessoal", "galerinha", "acho", "daí", "porta", "hein", "bora", 
    "aham", "juma", "tipo", "então", "assim", "agora", "tudo", "todos",
    "oi", "e aí", "eae", "fala", "galera", "bem-vindos", "cara", "meu", 
    "nossa", "deus", "jeito", "ideia", "pronto", "vocês", "ufa", "desculpa", 
    "cuidado", "socorro", "ola",
    
    # Verbos de Ação/Modalidade
    "posso", "pode", "podemos", "tenho", "tem", "temos", "ter",
    "vou", "vai", "vamos", "ir", "fui", "foi", "quero", "quer", "queria",
    "conseguir", "consegui", "consigo", "fazer", "faço", "fez",
    "pegar", "peguei", "pega", "usar", "uso", "botar", "colocar",
    "voltar", "sou", "é", "era", "preciso", "precisar", "precisa",
    "estou", "está", "tava", "estava", "estar", "ficar", "ficando", "ficou",
    "chegar", "cheguei", "chega", "acabar", "acabou", "ver",
    
    # Youtube / Engajamento / Alucinações
    "inscreva", "canal", "like", "vídeo", "video", "sininho",
    "notificação", "compartilha", "deixa", "gostei", "comenta",
    "subtitles", "subtitle", "caption", "captions", "watching", 
    "thanks", "thank", "you", "amara", "org", "community", "music", 
    "applause", "subscribe", "by", "transcription", "transcrição",
    
    # Dêiticos (Apontamentos)
    "lá", "aí", "ali", "cá", "já", "depois", "esse", "essa", "isso", 
    "esse daqui", "desse", "dessa", "disso", "aquele", "aquela", "aquilo",
    "nessa", "nesse", "nisso", "num", "numa", "onde", "quando", "como", 
    "porque", "por", "para", "here",

    # Domínio de jogos
    "minecraft", "jogo", "games", "baú", "bau", "chão", "teto", 
    "buraco", "alavanca", "mapa", "bloco",

    # Interjeições e gírias
    "caramba", "vixe", "eita", "uou", "amigos", "amigo", "cara", 
    "caraca",
    
    # Genéricos (substantivos vazios)
    "coisa", "coisas", "parte", "vez", "hora", "momento",
    "mundo", "lado", "frente", "cima", "baixo", "lugar",
    "sorte", "pouquinhos", "carinhas", "monte", "meio",
    "cada", "novo", "tempo", "itens", "fim", "vida",
    "vezes", "vez", "coisa", "coisas", "parte", "hora",
    "jeito", "lado", "fundo", "túnel", "busca", "vontade",
    "bom", "boa", "melhor", "comigo", "contigo", "mim",
    "lado", "pouco", "fio", "tábua", "verdade", "certeza",
    "favor", "super", "presente", "ovo", "tela", "menos",
    "vídeo", "videos", "youtube", "canal", "barulho", 
    "voz", "som", "música", "moto", "carro",

    # Interação entre pessoas
    "spock", "pokao", "pokão", "iago", "iagao", "iagão",
    "pocão", "bella", "kai", "cadres", "eduardo", "juju",
    "lulú", "amandinha", 

    # Vocativos e Gírias Específicas
    "moço", "moco", "amiguinho", "amiguinhos", "cara", "caras", 
    "pessoal", "gente", "filha", "pai", "menina", "menino",
    "meninas", "meninos", "moça", "vovô", "rapaz", "mames", "mami",
    "senhora", "senhor",


    # Interjeições e Ruído
    "uhul", "uhuu", "oba", "vixe", "nossa", "socorro", "ai",
    "likes", "like", "haha", "uhuuu", "wow", "uiuu", "uuuu",
    "uau", "hey", "huh", 
    
    # Verbos que parecem substantivos ou escaparam
    "vamo", "vamos", "pulo", "olha", "visto",

    # Erros do Spacy / Verbos disfarçados
    "ixi", "viu", "pula", "bota", "tadin", "tadinho",
    
    # Meta e Genéricos
    "videos", "video", "ano", "pessoas", "sala",
    
    # Humor / Específicos repetitivos (opcional)
    "pum", "bolinha",

    # Inglês
    "the", "videos", "youtube", "youtuber", "youtubers", "here", 
    "this", "can", "block", "get", "but", "rock", "pig", "pigs", 
    "piggy", "bot", "doll", "missy", "guys", "everyone", "liked",
    "number", "channel", "your", "bunny", "teacher", "now",
    "show", "people", "comment", "favorite", "friends", "choose",
    "sister", "minnie",

    # Verbos que o Spacy errou
    "morri", "caí", "cai", "rimarão",

    # Diminutivos e aumentativos
    "pouquinhos", "pouquinho", "carinhas", "cantinho", 
    "canto", "ladinho", "escadinha", "lobinho", "bichão",
    "lugarzinho", "bloquinhos", "casinha", "foguinho",
    "anjinho", "juntinho", "carrinho", "trenzinho", "salgadinho",
    "professorzinho", "mocinho", "patinho", "piscininha", "pezinho",
    "menininho", "joguinho", "dinheirinho", "bonitozinhozinho",
    "bonitinhozinhozinho", "mercadinho", "miguelinho",
    "cunezinho", "mandinha", "vazinho", "puxadinha", "mozinho",
    "fogãozinho", "gruzinho", "balãozinho", "motinha", "ônibusinho",
    "macarrãozinho", "quartinho", "patinhos", "olhinho", 
    "bebezinho", "cachorrrinho", "caixinha", "ppzinho",

    # Plurais
    "blocos", "armaduras", "espadas", "mobs", "poções",
    "carros", "motos", "vídeos", "alunos", "estudantes",
    "fones", "personagens", "brinquedos",
    
    # Pronomes que escaparam
    "você", "mim", "comigo",

    # Ruído
    "dwi", "gwybod", "ddechrau", "ddod", "nam", "tui",
    "fawr", "siarad", "byddwn", "wneud", "rydym", "mena",
    "zúúúúca", 
}

# --- Intervalos de otimização ---
param_ranges = {
    "n_neighbors": {"type": "int", "low": 5, "high": 10},
    "n_components": {"type": "int", "low": 2, "high": 5},
    "min_dist": {"type": "float", "low": 0.0, "high": 0.5},
    "min_cluster_size": {"type": "int", "low": 7, "high": 10},
    "min_samples": {"type": "int", "low": 3, "high": 30},
    "min_df": {"type": "int", "low": 2, "high": 15}, # Inteiro para limpeza absoluta
    "max_df": {"type": "float", "low": 1.0, "high": 1.0},
    # "ngram_range": {"type": "categorical", "choices": [(1,1), (1,2), (1,3)]},
    "ngram_range": {"type": "categorical", "choices": [(1,1)]},
}

'''
    Menu para selecionar qual conjunto de dados analisar
'''
def escolher_grupo():    
    # Monta lista de opções: Geral + Categorias Únicas + Youtubers
    categorias = sorted(list(set(MAPA_YOUTUBERS_CATEGORIA.values())))
    youtubers = sorted(list(MAPA_YOUTUBERS_CATEGORIA.keys()))
    
    opcoes = ["Geral"] + [f"[Cat] {c}" for c in categorias] + [f"[Ytb] {y}" for y in youtubers]
    
    pergunta = [
        {
            "type": "list",
            "name": "grupo",
            "message": "Qual grupo você deseja analisar?",
            "choices": opcoes,
        }
    ]
    resposta = prompt(pergunta)["grupo"]
    
    # Limpa a string para pegar o nome real (remove [Cat] ou [Ytb])
    if resposta.startswith("[Cat] "):
        return resposta.replace("[Cat] ", "")
    elif resposta.startswith("[Ytb] "):
        return resposta.replace("[Ytb] ", "")
    return resposta

'''
    Função para definir a sequência de passos da análise de tópicos com o BERTopic
    @param grupo_selecionado - Nome do grupo a ser analisado (Ex: Geral, Minecraft, Roblox, ...)
'''
def pipeline_BERTopic(grupo_selecionado):
    console.rule(f"[bold magenta]Pipeline BERTopic: {grupo_selecionado}[/bold magenta]")
    
    # Coleta de dados (com ou sem filtro gramatical)
    documentos = get_dados(grupo_analise=grupo_selecionado, usar_filtro_gramatical=True)
    
    if not documentos:
        console.print("[red]Nenhum documento encontrado para este grupo. Abortando.[/red]")
        return

    # Otimização Bayesiana dos parâmetros do BERTopic
    console.print(f"[green]Iniciando otimização com {len(documentos)} documentos...[/green]")
    study = otimizar_BERTopic(documentos, param_ranges, stop_words=list(custom_sWords), n_trials=15)
    
    # Salvamento do resultado
    if study and study.best_value > -1: # Aceita 0.0 como válido, mas não erro
        console.print(f"[bold blue]Melhor score:[/bold blue] {study.best_value}")
        console.print(f"[bold blue]Parâmetros:[/bold blue] {study.best_params}")
        
        # Passa o nome do grupo como parâmetro para criar a subpasta correta
        salvar_BERTopic(documentos, study.best_params, stop_words=list(custom_sWords), nome_grupo=grupo_selecionado)
    else:
        # Tenta salvar mesmo se o score for baixo, para debug, se houver params
        if study:
             console.print("[yellow]Score baixo, mas salvando melhor tentativa encontrada...[/yellow]")
             salvar_BERTopic(documentos, study.best_params, stop_words=list(custom_sWords), nome_grupo=grupo_selecionado)
        else:
             console.print("[red]Falha crítica na otimização.[/red]")

'''
    Função interativa para ajustar ranges do Optuna
'''
def editar_parametros():
    global param_ranges
    print("\n--- Ranges atuais ---")

    for k, v in param_ranges.items():
        if v["type"] != "categorical":
            print(f"{k}: [{v['low']} , {v['high']}]")
        else:
            print(f"{k}: {v['choices']}")

    print("----------------------\n")

    if not prompt({"type": "confirm", "name": "c", "message": "Alterar parâmetros?", "default": False})["c"]:
        return

    for param, cfg in param_ranges.items():
        if cfg["type"] == "categorical":
            continue

        q = [
            {"type": "input", "name": "low", "message": f"Min {param} ({cfg['low']}):"},
            {"type": "input", "name": "high", "message": f"Max {param} ({cfg['high']}):"}
        ]
        ans = prompt(q)
        
        try:
            val_low = float(ans["low"])
            val_high = float(ans["high"])
            if cfg["type"] == "int":
                param_ranges[param]["low"] = int(val_low)
                param_ranges[param]["high"] = int(val_high)
            else:
                param_ranges[param]["low"] = val_low
                param_ranges[param]["high"] = val_high
        except:
            print("Valor inválido, mantendo anterior.")
    print("✔ Atualizado.")

'''
    Menu principal do pipeline do BERTopic
'''
def main_menu():
    while True:
        menu = [
            {
                "type": "list",
                "name": "acao",
                "message": "Selecione uma opção:",
                "choices": [
                    "1. Executar Pipeline (Treinar)", 
                    "2. Visualizar Modelo Existente", 
                    "3. Editar Parâmetros", 
                    "Sair"
                ],
            }
        ]
        choice = prompt(menu)["acao"]

        if "Sair" in choice:
            break

        elif "3. Editar" in choice:
            editar_parametros()

        elif "1. Executar" in choice:
            grupo = escolher_grupo()
            pipeline_BERTopic(grupo)

        elif "2. Visualizar" in choice:
            grupo = escolher_grupo()
            visualizar_bertopic(nome_grupo=grupo)

if __name__ == "__main__":
    main_menu()