from transformers import AutoTokenizer
from detoxify import Detoxify
from rich.console import Console

console = Console()

def testar_barreira_tokens():
    console.print("[bold]Iniciando Teste de Truncamento de 512 Tokens...[/bold]")

    # 1. Carregar o tokenizador exato usado pelo Detoxify Multilingual
    console.print("Carregando tokenizador XLM-RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # 2. Criar uma base neutra. 
    # A palavra "legal" e o espaço " " formam 1 token exato na maioria dos casos.
    # Multiplicamos por 510 para deixar espaço para os tokens especiais <s> e </s>.
    palavra_base = "legal "
    texto_510_palavras = palavra_base * 510

    # Vamos garantir que temos exatamente 510 tokens úteis decodificando um corte exato
    # encode() transforma em números, [1:-1] remove o <s> e </s> temporariamente
    tokens_brutos = tokenizer.encode(texto_510_palavras)[1:-1]
    
    # Pegamos exatamente os primeiros 510 tokens e transformamos de volta em texto
    tokens_exatos = tokens_brutos[:510]
    trecho_limpo = tokenizer.decode(tokens_exatos)
    
    # 3. Construir os cenários
    # Cenário A: Exatamente 512 tokens (1 <s> + 510 palavras + 1 </s>)
    # Cenário B: Exatamente 513 tokens (1 <s> + 510 palavras + 1 palavrão + 1 </s>)
    
    palavrao = " merda" # Adicionamos um espaço antes para não colar na última palavra
    trecho_sujo = trecho_limpo + palavrao

    # 4. Verificação de Sanidade Matemática
    tam_limpo = len(tokenizer.encode(trecho_limpo))
    tam_sujo = len(tokenizer.encode(trecho_sujo))
    
    console.print(f"\n[cyan]Verificação de Tamanho:[/cyan]")
    console.print(f"Tamanho do Trecho Neutro: {tam_limpo} tokens")
    console.print(f"Tamanho do Trecho com Palavrão: {tam_sujo} tokens")

    if tam_limpo != 512 or tam_sujo != 513:
        console.print("[red]Atenção: A contagem de tokens não está exata. Reveja o tokenizador.[/red]")

    # 5. O Teste de Fogo com o Detoxify
    console.print("\n[cyan]Rodando Inferência no Detoxify...[/cyan]")
    modelo = Detoxify('multilingual')
    
    score_limpo = modelo.predict(trecho_limpo)['toxicity']
    score_sujo = modelo.predict(trecho_sujo)['toxicity']

    console.print(f"\n[bold green]RESULTADOS:[/bold green]")
    console.print(f"Score exato (512 tokens): {score_limpo:.4f}")
    console.print(f"Score estourado (513 tokens): {score_sujo:.4f}")
    
    if abs(score_limpo - score_sujo) < 0.0001:
        console.print("\n[bold yellow]CONCLUSÃO: O score não mudou! O palavrão na posição 513 foi sumariamente ignorado.[/bold yellow]")
    else:
        console.print("\n[bold red]CONCLUSÃO: O score mudou. O modelo leu o palavrão de alguma forma.[/bold red]")

if __name__ == "__main__":
    testar_barreira_tokens()