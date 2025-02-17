config = {
  # Configuração da região da coleta -> Formato: ISO 3166-1 alpha-2
  'region_code': 'BR',         

  # Configuração da linguagem da coleta -> Formato: ISO 639-1   
  'relevance_language': 'pt',     

  # A coleta ocorre da data final para a data inicial -> [ano, mês, dia]
  'start_date': [2020, 1, 1], 
  'end_date': [2025, 1, 10],

  # API que receberá uma requisição PATCH com payload de um JSON contendo informações acerca da coleta
  # Mantenha uma string vazia '' Caso não tenha configurado
  'api_endpoint': '',
  # Intervalo, em segundos, entre cada envio de dados para a API
  'api_cooldown': 60,                                                   

  # Intervalo, em segundos, entre cada tentativa de requisição para a API apos falha
  'try_again_timeout': 60,                                              

  # Palavras que serão utilizadas para filtrar os títulos dos vídeos
  'key_words': [
      'minecraft',
      'Minecraft'],

  # KEYs da API v3 do YouTube
  'youtube_keys': [
    'AIzaSyBa7xnlhKulIqvh15fNgRXnJWIOMLzxYs8',
    'AIzaSyB_nwBTSmNfVP2HyyJzc4NMKgAEFX-lJk0',
    'AIzaSyDpOjnhCWpAbRKWsDI7mVqufzlMO4DNHUI',
    'AIzaSyDx1kErv6e02FygrSgdYIrCmkWw4IIXZlU',
    'AIzaSyCg9e0EeIEM_hE9LmjCvoo6qNQrHN1raz8',
    'AIzaSyCuAKpgB5Q38PUbIrstFfU5ZQEr4xb13q0',
    'AIzaSyDy3ioPkbzXVQ2EaDDjGGc3vYgxCIBtrJg',
    'AIzaSyCmyWHJ7UM9i2Vb6_X3QSy1D_QgAlkY3Oo',
    'AIzaSyBlnJKHNU3Z_7UUmFmzRWVxVjldOQNkrvw'
  ],


  # Queries que serão utilizadas na pesquisa
'queries': [
      'minecraft',
      'Minecraft',
      'mine'
],

# ID do canal para restringir a busca
    'channel_id': [
        'UCIPA6iWNaoetaa1T46RkzXw', #Autentic Games
        'UCtKndmEnQyqkhOQfpi5CgvQ', #Geleia
        'UCYtaoJSk9iIrFxx8eZB15ag', #Robin Hood
        'UCIXguhHCl8eDTkXpEuiGPUA', #Jazzgost
        'UCfqTbhALHYGLa20p2qXrdzw', #Cherryrar
        'UCYCnGi1DXHvHlffFY7pEDmQ', #Minguado
        'UCjBO43ykxlSs3j7F7EXcBUQ', #Tazercraft
        'UCV306eHqgo0LvBf3Mh36AHg'  #Felipe Neto
    ], 

# Grau onde considera o valor positivo, neutro ou negativo na analise de sentimento
 'treshold':[0.5, 0.5, 0.5]
}

