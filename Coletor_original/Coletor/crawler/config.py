config = {
  # Configuração da região da coleta -> Formato: ISO 3166-1 alpha-2
  'region_code': 'BR',         

  # Configuração da linguagem da coleta -> Formato: ISO 639-1   
  'relevance_language': 'pt',     

  # A coleta ocorre da data final para a data inicial -> [ano, mês, dia]
  'start_date': [2023, 7, 1], 
  'end_date': [2024, 8, 1],

  # API que receberá uma requisição PATCH com payload de um JSON contendo informações acerca da coleta
  # Mantenha uma string vazia '' Caso não tenha configurado
  'api_endpoint': '',
  # Intervalo, em segundos, entre cada envio de dados para a API
  'api_cooldown': 60,                                                   

  # Intervalo, em segundos, entre cada tentativa de requisição para a API apos falha
  'try_again_timeout': 60,                                              

  # Palavras que serão utilizadas para filtrar os títulos dos vídeos
  'key_words': [
      'Ministro Haddad', 'Ministro Fernando Haddad',
      'Fernando Haddad', 'Haddad'
      'Ministro da Fazenda', 
      'Ministério da Fazenda'],

  # KEYs da API v3 do YouTube
  'youtube_keys': [
    'AIzaS XXXX XXXXX'
  ],

  # Queries que serão utilizadas na pesquisa
'queries': [
      'Ministro Haddad', 'Ministro Fernando Haddad',
      'Haddad', 'Fernando Haddad',
      'Ministro da Fazenda', 
      'Ministério da Fazenda',
      'Economia do país'
]
}