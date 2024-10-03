from datetime import datetime

def log(type, message):
  now = datetime.now()
  now_str = "{:02d} {:02d}:{:02d}:{:02d}".format(now.day, now.hour, now.minute, now.second)

  typesList = ["error", "key", "video", "search", "channels"]
  # typesList = []
  if type in typesList: print(f"{now_str} ({type}) - {message}")