from detoxify import Detoxify

textos_para_analise = []

textos_para_analise.append("Senão você vai ser eliminado se você morrer.")

textos_para_analise.append("Tá. Será que a gente vai aqui no meio da floresta? Peraí, fica atrás das árvores. Tá, tô atrás. Ai, meu Deus. Não faz barulho. Não, tô medir. Ele pode estar por perto. Ah, usa o rastrador, peraí. Boa. Rastrador. Parece que não. Não funciona. Não, parece que ele não tá perto. Nossa, cadê? Aqui, ó. Atira, atira. Toma, toma. Nossa, ele vai morrer, Cris. Vai, vai, vem. Tá acontecendo comigo. Não, não, não. Enquanto ele tá indo atrás de você, continue nessa direção.")

textos_para_analise.append("Atira nele aqui. Tô atirando, tô atirando. Tô atirando, vai. Ele tá menos da metade da vida. Eletá correndo. Vanta aqui, seu medruzão. Aí, ó. Nossa, ele vai morrer, Cris. Vai morrer, vai morrer. Atira, Crisinha")

textos_para_analise.append("Aí, ó")

try:
    detoxify_model = Detoxify('multilingual')

    resultados = detoxify_model.predict(textos_para_analise)

    print(resultados)
except Exception as e:
    print("Inválido")