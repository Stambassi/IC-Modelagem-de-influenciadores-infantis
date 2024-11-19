config = {
  # Configuração da região da coleta -> Formato: ISO 3166-1 alpha-2
  'region_code': 'BR',         

  # Configuração da linguagem da coleta -> Formato: ISO 639-1   
  'relevance_language': 'pt',     

  # A coleta ocorre da data final para a data inicial -> [ano, mês, dia]
  'start_date': [2019, 1, 1], 
  'end_date': [2021, 12, 31],

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
    'AIzaSyDy3ioPkbzXVQ2EaDDjGGc3vYgxCIBtrJg'
  ],


  # Queries que serão utilizadas na pesquisa
'queries': [
      'minecraft',
      'Minecraft',
      'mine'
],

# ID do canal para restringir a busca
    'channel_id': [
        'UCIPA6iWNaoetaa1T46RkzXw',
        'UCEfGV5hx2VrXl4jOUnG0MRQ',
        'UCtKndmEnQyqkhOQfpi5CgvQ'         
    ], 
}