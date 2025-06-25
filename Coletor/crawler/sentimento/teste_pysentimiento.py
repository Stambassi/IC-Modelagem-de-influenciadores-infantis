from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="pt")

resultados = []

resultados.append(analyzer.predict("Vem com a picareta e começa a bater nela assim, ó, com o botão direito, ó. Vai batendo, Iagão").probas)

resultados.append(analyzer.predict("Você é velha igual minha vó").probas)

print(resultados)
