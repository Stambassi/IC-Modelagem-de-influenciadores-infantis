from transformers import AutoTokenizer
from detoxify import Detoxify
from rich.console import Console
from rich.table import Table

console = Console()

'''
    Função para testar o truncamento e limite de contexto arquitetural
    
    @param modelo - Instância carregada do modelo Detoxify
'''
def testar_barreira_tokens(modelo, tokenizer):
    console.print("\n[bold magenta]--- TESTE 1: LIMITES DE TOKENS (512 vs 513) ---[/bold magenta]")
    
    palavra_base = "legal "
    texto_510_palavras = palavra_base * 510

    # Extração de exatos 510 tokens limpos
    tokens_brutos = tokenizer.encode(texto_510_palavras)[1:-1]
    tokens_exatos = tokens_brutos[:510]
    trecho_limpo = tokenizer.decode(tokens_exatos)
    
    # Adição de um token extra sujo
    palavrao = " merda"
    trecho_sujo = trecho_limpo + palavrao

    tam_limpo = len(tokenizer.encode(trecho_limpo))
    tam_sujo = len(tokenizer.encode(trecho_sujo))
    
    console.print(f"[cyan]Verificação:[/cyan] Neutro: {tam_limpo} tokens | Com Palavrão: {tam_sujo} tokens")

    score_limpo = modelo.predict(trecho_limpo)['toxicity']
    score_sujo = modelo.predict(trecho_sujo)['toxicity']

    console.print(f"Score exato (512 tokens): {score_limpo:.4f}")
    console.print(f"Score estourado (513 tokens): {score_sujo:.4f}")
    
    if abs(score_limpo - score_sujo) < 0.0001:
        console.print("[bold yellow]CONCLUSÃO: O score não mudou. Truncamento silencioso confirmado![/bold yellow]")
    else:
        console.print("[bold red]CONCLUSÃO: O score mudou. Revisar tokenização.[/bold red]")

'''
    Função para avaliar a sensibilidade do modelo quanto à posição e frequência de termos tóxicos dentro de uma janela válida de SVT
    
    @param modelo - Instância carregada do modelo Detoxify
'''
def testar_posicao_frequencia(modelo):
    console.print("\n[bold magenta]--- TESTE 2: POSIÇÃO E FREQUÊNCIA (EFEITO DILUIÇÃO) ---[/bold magenta]")
    
    # Simulando um SVT de ~1 minuto (100 palavras)
    N = 100
    palavra_base = "legal"
    palavra_toxica = "merda"
    
    # Função auxiliar para construir o texto injetando a palavra tóxica em posições específicas
    def construir_texto(posicoes):
        palavras = [palavra_base] * N
        for p in posicoes:
            palavras[p] = palavra_toxica
        return " ".join(palavras)

    # Dicionário com os cenários desenhados para a pesquisa
    cenarios = {
        "Base (0 tóxicas)": [],
        "1 Tóxica (Início do SVT)": [0],
        "1 Tóxica (Meio do SVT)": [N // 2],
        "1 Tóxica (Fim do SVT)": [N - 1],
        "3 Tóxicas (Espalhadas)": [10, 50, 90],
        "5 Tóxicas (Espalhadas)": [10, 30, 50, 70, 90],
        "5 Tóxicas (Concentradas)": [50, 51, 52, 53, 54]
    }

    # Estruturação visual da tabela
    tabela = Table(title="Impacto no SVT de 1 minuto (100 tokens)")
    tabela.add_column("Cenário", style="cyan")
    tabela.add_column("Score", justify="right")
    tabela.add_column("Classificação", justify="center")

    for nome, posicoes in cenarios.items():
        texto = construir_texto(posicoes)
        score = modelo.predict(texto)['toxicity']
        
        # Aplicação direta da sua discretização metodológica
        if score < 0.20:
            classe = "[bold green]NT[/bold green]"
        elif score < 0.80:
            classe = "[bold yellow]GZ[/bold yellow]"
        else:
            classe = "[bold red]T[/bold red]"
            
        tabela.add_row(nome, f"{score:.4f}", classe)

    console.print(tabela)
    console.print("\n[dim]Nota: 'Espalhadas' simula xingamentos pontuais ao longo de 1 minuto. "
                  "'Concentradas' simula uma explosão de raiva (rage) contínua.[/dim]")

'''
    Função para avaliar a sensibilidade do modelo com textos reais
    Simula transcrições reais de gameplays de Minecraft/Roblox (aprox. 1 minuto de fala),
    variando a posição e a densidade das toxicidades (rage) em um discurso natural
    
    @param modelo - Instância carregada do modelo Detoxify
'''
def testar_textos_reais(modelo):
    console.print("\n[bold magenta]--- TESTE 3: TEXTOS REAIS E VALIDADE ECOLÓGICA ---[/bold magenta]")
    
    # Textos simulando a transcrição do Whisper para um vídeo típico de Roblox/Minecraft
    cenarios_reais = {
        "Base (0 tóxicas)": 
            "E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, cuidado com a lava. Boa, passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam!",
            
        "1 Tóxica (Início do SVT)": 
            "Que caralho, já começou travando! E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, cuidado com a lava. Boa, passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam!",
            
        "1 Tóxica (Meio do SVT)": 
            "E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, caí na lava, puta que pariu. Boa, voltamos e passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam!",
            
        "1 Tóxica (Fim do SVT)": 
            "E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, cuidado com a lava. Boa, passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam! Morri no finalzinho, vai se ferrar.",
            
        "3 Tóxicas (Espalhadas)": 
            "Que caralho, já começou travando! E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, caí na lava, puta que pariu. Boa, voltamos e passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like e se inscrevam! Morri no final, vai se ferrar.",
            
        "5 Tóxicas (Espalhadas)": 
            "Que caralho, começou travando! E aí pessoal, tudo bem? Começando mais uma partida aqui. Hoje vamos jogar no modo sobrevivência. Essa merda de controle não ajuda. Chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja. Vamos pular por aqui, caí na lava, puta que pariu. Boa, voltamos e passamos! Agora é só abrir a porra do baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam! Morri no final, vai se ferrar.",
            
        "5 Tóxicas (Concentradas)": 
            "E aí pessoal, tudo bem? Começando mais uma partida aqui no canal. Hoje vamos jogar o modo sobrevivência e tentar bater o nosso recorde. Eu chamei alguns amigos para ajudar na missão. Olha só, peguei uns itens novos na loja que vão dar uma vantagem legal. Vamos pular por aqui, cuidado com a lava. Puta que pariu, caralho, merda, vai se ferrar, desgraça de jogo! Boa, voltamos e passamos! Agora é só abrir o baú e ver o que ganhamos. Deixem o like no vídeo e se inscrevam!"
    }

    # Estruturação visual da tabela
    tabela = Table(title="Impacto de Rage em Transcrições Reais (SVT ~1 min)")
    tabela.add_column("Cenário", style="cyan")
    tabela.add_column("Score", justify="right")
    tabela.add_column("Classificação", justify="center")

    for nome, texto in cenarios_reais.items():
        score = modelo.predict(texto)['toxicity']
        
        # Discretização conforme a metodologia definida
        if score < 0.20:
            classe = "[bold green]NT[/bold green]"
        elif score < 0.80:
            classe = "[bold yellow]GZ[/bold yellow]"
        else:
            classe = "[bold red]T[/bold red]"
            
        tabela.add_row(nome, f"{score:.4f}", classe)

    console.print(tabela)

if __name__ == "__main__":
    console.print("[bold]Inicializando ambiente e carregando modelos...[/bold]")
    tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
    modelo_detoxify = Detoxify('multilingual')
    
    # Executa a suíte de testes completa
    testar_barreira_tokens(modelo_detoxify, tokenizer_xlm)
    testar_posicao_frequencia(modelo_detoxify)
    testar_textos_reais(modelo_detoxify) # A nova chamada