from datetime import datetime

class GlobalState:
  _instance = None
  _work_start = None
  _work_end = None

  _state = {}

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(GlobalState, cls).__new__(cls, *args, **kwargs)
    return cls._instance

  def __init__(self):
    pass

  @staticmethod
  def get_instance() -> "GlobalState":
    if GlobalState._instance is None:
      GlobalState._instance = GlobalState()
    return GlobalState._instance

  def _update_time_running(self):
#    diff = abs(self._state["last_start"] - self._state["last_sleep"] )
    last_start = datetime.fromisoformat(self._state["last_start"])
    last_sleep = datetime.fromisoformat(self._state["last_sleep"])
    diff = abs(last_start - last_sleep)

    seconds = diff.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    str_hour = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    self._state["time_running"] = str_hour

  def set_state(self, key, value):
    self._state[key] = value

    if key == "status" and value == "working":
      self._work_start = datetime.now()
      self._state["last_start"] = self._work_start.isoformat()

    if key == "status" and value == "sleeping":
      self._work_end = datetime.now()
      self._state["last_sleep"] = self._work_end.isoformat()
      self._update_time_running()

    # print(self._state)

  def get_state(self):
    return self._state
