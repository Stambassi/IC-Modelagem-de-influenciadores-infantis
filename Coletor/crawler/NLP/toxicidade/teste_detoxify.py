from detoxify import Detoxify

textos_para_analise = []

textos_para_analise.append("Quem tem um olho em mim? Ela consegue ver onde a gente está Acho que ela consegue ver onde a gente está Ela identifica nossa posição Ela está aqui Ela morreu Tem que ver alguém Nossa, ela está fazendo uma festa ali em todo mundo No camping Você não pode cair, por favor No camping Vem pra cá, Cris, você morreu Não, eu já estou caída no chão Ela fica camperando Eu gibrei, eu gibrei, eu gibrei ela Ela morreu Ela morreu Reviveu todo mundo Todo mundo junto, né? E alguém já morreu pra valer Sério? Eu só tenho uma pergunta pra vocês A gente só tem que ficar vivo, essa é a única missão Sim, mas está complicado Ela está aqui, ela derrubou alguém Estou vendo, estou vendo Calma, cuidado ali perto")

textos_para_analise.append("Quem tem um olho em mim? Ela consegue ver onde a gente está Acho que ela consegue ver onde a gente está Ela identifica nossa posição Ela está aqui Ela morreu Tem que ver alguém Nossa, ela está fazendo uma festa ali em todo mundo caralho No camping Você não pode cair, por favor No camping Vem pra cá, Cris, você morreu Não, eu já estou caída no chão Ela fica camperando Eu gibrei, eu gibrei, eu gibrei ela Ela morreu Ela morreu Reviveu todo mundo Todo mundo junto, né? E alguém já morreu pra valer Sério? Eu só tenho uma pergunta pra vocês A gente só tem que ficar vivo, essa é a única missão Sim, mas está complicado Ela está aqui, ela derrubou alguém Estou vendo, estou vendo Calma, cuidado ali perto")

try:
    detoxify_model = Detoxify('multilingual')

    resultados = detoxify_model.predict(textos_para_analise)

    print(resultados)
except Exception as e:
    print("Inválido")