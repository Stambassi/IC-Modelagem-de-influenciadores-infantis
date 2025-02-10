import datetime

def secondsUntil(n):
  now = datetime.datetime.now()
  target = now.replace(hour=n, minute=0, second=0, microsecond=0)

  if now.hour >= n:
    target = target + datetime.timedelta(days=1)

  time_remaining = target - now

  seconds_remaining = int(time_remaining.total_seconds())
  return seconds_remaining