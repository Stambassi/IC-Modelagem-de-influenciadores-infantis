from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="pt")

resultados = []

resultados.append(analyzer.predict("eita").probas)

resultados.append(analyzer.predict("eita porra").probas)

resultados.append(analyzer.predict("porra").probas)

resultados.append(analyzer.predict("Eita, que coisa doida").probas)

resultados.append(analyzer.predict("Ahhh, entendi nada").probas)

resultados.append(analyzer.predict("entendi nada").probas)

resultados.append(analyzer.predict("Ahhh").probas)

print(resultados)
