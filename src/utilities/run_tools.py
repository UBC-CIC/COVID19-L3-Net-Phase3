from datetime import datetime, timedelta
import os
import re
from math import ceil
from time import sleep
from tqdm import tqdm
import sys


def timestamp(fmt=r'[%Y/%m/%d @ %H:%M:%S] '):
    now = datetime.now()
    return now.strftime(fmt)

class Printer(object):
    def __init__(self, log_path, filename, mode='a') -> None:
        self.file = open(os.path.join(log_path, filename), mode=mode)
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.file.write(message)
        self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    

class RunTimer:
    def __init__(self, log_path, name) -> None:
        self.time_points = []
        self.comments = []
        self.levels = []
        self.log_path = log_path
        self.name = name

    def add_marker(self, comment='', level=0):
        self.time_points.append(datetime.now())
        self.comments.append(comment)
        self.levels.append(level)

    def summary(self, fmt=r'%Y/%m/%d @ %H:%M:%S'):
        file = open(os.path.join(self.log_path, 'runtime_markers.log'), 'a')
        print("{} Run Summary: ".format(self.name), file=file)
        print("{} Run Summary: ".format(self.name), file=None)
        for idx, dtobj in enumerate(self.time_points):
            print("\t" + "\t" * self.levels[idx] + dtobj.strftime(fmt) + " - " + self.comments[idx], file=file, flush=True)
            print("\t" + "\t" * self.levels[idx] + dtobj.strftime(fmt) + " - " + self.comments[idx], file=None, flush=True)
        file.close

    def exit_handler(self):
        self.add_marker('Stop')
        self.summary()


class RunScheduler:
    def __init__(self, start_time, check_frequency = 5) -> None:
        self.start_time = start_time
        self._parse_time()
        print("Run is scheduled for: {}. Sleeping now...".format(self.start_time.strftime(r"%Y-%m-%d %H:%M:%S")), flush=True)
        while datetime.now() < self.start_time:
            sleep(check_frequency * 60)

        print("Waking up! Resuming scheduled run...")

    def _parse_time(self):
        assert isinstance(int(self.start_time), int)

        if len(self.start_time) == 4: # HHMM
            now = datetime.now()
            start = datetime.strptime(self.start_time,'%H%M').replace(year=now.year, month=now.month, day=now.day)
            if (start-now).days < 0:
                start = start.replace(day=now.day + 1)

        if len(self.start_time) == 6: # ddHHMM
            now = datetime.now()
            start = datetime.strptime(self.start_time, r"%d%H%M").replace(year=now.year, month=now.month)
            if (start-now).days < 0:
                start = start.replace(month=now.month + 1)
            
        if len(self.start_time) == 8: # mmddHHMM
            now = datetime.now()
            start = datetime.strptime(self.start_time, r"%m%d%H%M").replace(year=now.year)
            if (start-now).days < 0:
                start = start.replace(year=now.year + 1)

        if len(self.start_time) == 12: # yyyymmddHHMM
            now = datetime.now()
            start = datetime.strptime(self.start_time, r"%Y%m%d%H%M")
            if (start-now).days < 0:
                raise ValueError("Scheduled start cannot be in the past!")
        
        self.start_time = start


