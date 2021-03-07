from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import copy
import time
import queue
import json, itertools, collections
import pickle
import traceback
import multiprocessing
from functools import partial
from functools import total_ordering
from multiprocessing import Process, Manager

from run_glue import args, main

_args = args
_sys_stdout = sys.stdout

class EndSignal(object):
  pass

class ChoiceParameter(object):
  def __init__(self, name, values):
    self._name = name
    self._values = values

  @property
  def name(self):
    return self._name

  def __iter__(self):
    return iter(self._values)

  @property
  def setting_count(self):
    return len(self._values)

  def __str__(self):
    return "Parameter '{}': {}".format(self._name, self._values)

class ParameterManager(object):
  def __init__(self):
    self._supported_param_types = {
        "choice": self._choice_param
    }

  def from_dict(self, param_desc):
    if "type" not in param_desc:
      raise ValueError("Invalid parameter description {}, "
                       "'type' missing.".format(param_desc))
    if param_desc["type"] not in self._supported_param_types:
      raise ValueError("Unsupported parameter type {}, "
                       "currently only '{}' supported.".format(
                           param_desc["type"],
                           self._supported_param_types.keys()))
    else:
      return self._supported_param_types[param_desc['type']](param_desc)

  def _choice_param(self, param_desc):
    if 'values' not in param_desc:
      raise ValueError("Invalid ChoiceParameter {}, "
                       "'values' missing.".format(param_desc))
    if not isinstance(param_desc['values'], collections.Iterable):
      raise ValueError("For ChoiceParameter, 'values' should be iterable. ")
    return ChoiceParameter(param_desc['name'], param_desc['values'])

class GridSearchTuner(object):
  def __init__(self):
    self._total_conf = 0
    self._generated_conf = 0
    self._params = {}
    self._param_manager = ParameterManager()

  def from_json(self, json_file):
    with open(json_file) as data_file:
      params = json.loads(data_file.read())
      conf_count = 1
      for param in params:
        if param['name'] in self._params:
          raise ValueError("Duplicated param {} found.".format(param['name']))
        param_obj = self._param_manager.from_dict(param)
        conf_count *= param_obj.setting_count
        self._params[param_obj.name] = param_obj

    self._total_conf = conf_count

    print("-" * 80)
    for key, value in self._params.items():
      print(value)

    print("")
    print("* Total conf count to try: {}".format(self._total_conf))

  @property
  def total_conf(self):
    return self._total_conf

  @property
  def generated_conf(self):
    return self._generated_conf

  @property
  def confs(self):
    param_names = self._params.keys()
    param_objs = self._params.values()

    for values in itertools.product(*param_objs):
      self._generated_conf += 1
      yield dict(zip(param_names, values))

class Logger(object):
  def __init__(self, output_file):
    self.log = open(output_file, "w")

  def write(self, message):
    self.log.write(message)
    self.log.flush()

class Job(object):
  def __init__(self, job_id, gpu_id, settings):
    self._job_id = job_id
    self._gpu_id = gpu_id
    self._settings = settings
    self.is_succeed = False
    self.exception = None
    self.ex_info = None
    self.result = 0.0
    self.all_result = {}
    self.log_file = None

  @property
  def settings(self):
    return self._settings

  @property
  def gpu_id(self):
    return self._gpu_id

  @property
  def job_id(self):
    return self._job_id

  def print_user_flags(self, flags, line_limit=80):
    print("-" * 80)
    for flag_name in dir(flags):
      value = "{}".format(getattr(flags, flag_name))
      log_string = "job_{:<5d}: ".format(self._job_id)
      log_string = flag_name
      log_string += "." * (line_limit - len(flag_name) - len(value))
      log_string += value
      print(log_string)
    print("-" * 80)

  def prepare(self):
    cur_process = multiprocessing.current_process()
    print("-" * 80)
    print("* Start to execute job {}".format(self._job_id))
    print("* GPU {} occupied by {}".format(self._gpu_id, cur_process.name))
    log_file = os.path.join(_args.output_dir, "job_{}.log".format(self._job_id))
    print("* Log file located at {}".format(log_file))
    print("* Settings: {}".format(self._settings))
    print("-" * 80)
    sys.stdout = Logger(log_file)
    self.log_file = log_file

  def execute(self):
    flags = copy.deepcopy(_args)
    cur_process = multiprocessing.current_process()
    flags.output_dir = os.path.join(flags.output_dir)
    #flags.output_dir = os.path.join(flags.output_dir, cur_process.name)
    flags.job_id = self._job_id
    print("flags.output_dir: ", flags.output_dir)
    self.update_flags(self._settings, flags)

    os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    self.print_user_flags(flags)
    self.result, self.all_result = main(flags)

  def update_flags(self, params, FLAGS):
    for k, v in params.items():
      assert hasattr(FLAGS, k), "Invalid parameter {} to update".format(k)
      flags_val_type = type(getattr(FLAGS, k))
      # if isinstance(getattr(FLAGS, k), str) and isinstance(v, unicode):
      #   v = v.encode("ascii")

      assert flags_val_type == type(v), \
              "Inconsistent type FLAGS: {} vs. {} - {}:{}".format(flags_val_type, type(v), k, v)
      setattr(FLAGS, k, v)

def evaluate_task(job_q, done_q, gpu_q, lock):
  while True:
    job = job_q.get()
    if isinstance(job, EndSignal):
      break;
    try:
      with lock:
        job.prepare()
      job.execute()
    except Exception as e:
      traceback.print_exc()
      job.is_succeed = False
      job.exception = str(traceback.format_exc())
      job.ex_info = type(e).__name__
    else:
      job.is_succeed = True
    finally:
      gpu_q.put(job.gpu_id)
      done_q.put(job)
      # write to log file
      if not job.is_succeed:
        print("job.log_file",job.log_file)
        with open(job.log_file, "a") as fout:
          fout.write("{}\n".format(job.exception))
      sys.stdout = _sys_stdout
      with lock:
        print("-" * 80)
        if job.is_succeed:
          print("* Job {} succeed, dev_acc: {:<.5f}".format(job.job_id, job.result))
        else:
          print("! Job {} failed".format(job.job_id))
          print("! Exception: {}".format(job.exception))
        print("* Release GPU {}".format(job.gpu_id))

  job_q.put(EndSignal())

if __name__ == '__main__':

  if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

  gpu_ids = _args.available_gpus
  num_task = int(len(gpu_ids.split(",")) / _args.need_gpus)
  print("Task Num: {}".format(num_task))

  tuner = GridSearchTuner()
  tuner.from_json(_args.conf_file)
  job_count = 0

  manager = Manager()
  job_q = manager.Queue(num_task)
  done_q = manager.Queue()
  gpu_q = manager.Queue(num_task)
  lock = manager.Lock()
  finished_jobs = []

  # fill gpu resources
  for i in range(num_task):
    gpu_id = ",".join(gpu_ids.split(",")[_args.need_gpus * i: _args.need_gpus * i + _args.need_gpus])
    gpu_q.put(gpu_id)

  workers = [
      Process(target=evaluate_task, args=(job_q, done_q, gpu_q, lock))
      for _ in range(num_task)
  ]

  for w in workers:
    w.daemon = True
    w.start()

  conf_iter = tuner.confs
  while True:
    try:
      gpu_id = gpu_q.get_nowait()
    except queue.Empty:
      time.sleep(0.01)
      new_done = False
      while not done_q.empty():
        finished_jobs.append(done_q.get())
        new_done = True
      if new_done:
        print("finished_jobs",finished_jobs)
        finished_jobs.sort(key=lambda t:t.result, reverse=True)
        with lock:
          print("-" * 80)
          print("* Progress: {}/{}".format(len(finished_jobs), tuner.total_conf))
          if finished_jobs[0].is_succeed:
            print("* Best Dev acc: {:<.5f}, Job: {}".format(finished_jobs[0].result,
                                                                     finished_jobs[0].job_id
                                                                    ))
            print("* Best settings: {}".format(finished_jobs[0].settings))
        # save to result_file
        result_file = os.path.join(_args.output_dir, "tune_results_"+args.task_name+"_dev.txt")
        with open(result_file, 'w') as fout:
          for job in finished_jobs:
            if job.is_succeed:
              fout.write("job_{}, dev_acc:{:<.5f}, {}\n".format(job.job_id, job.result,
                                                           job.settings))
              all_result_log = ""
              for key in sorted(job.all_result.keys()):
                all_result_log += " {}={:<.5f}".format(key, job.all_result[key])
              all_result_log += "\n"
              fout.write(all_result_log)

            else:
              fout.write("job_{}\t{}\t{}\n".format(job.job_id, job.ex_info, job.settings))
    else:
      try:
        conf = next(conf_iter)
        job = Job(job_count, gpu_id, conf)
        job_q.put(job)
        job_count += 1
      except StopIteration as e:
        break

  job_q.put(EndSignal())
  for w in workers:
    w.join()

  while not done_q.empty():
    finished_jobs.append(done_q.get())
  finished_jobs.sort(key=lambda t:t.result,reverse=True)
  print("-" * 80)
  print("* All job finished")
  if len(finished_jobs) > 0 and finished_jobs[0].is_succeed:
    print("* Best acc:{:<.5f}, Job: {}".format(finished_jobs[0].result, finished_jobs[0].job_id))
    print("* Best settings: {}".format(finished_jobs[0].settings))

  # save to result_file
  result_file=os.path.join(_args.output_dir,"tune_results_"+args.task_name+"_dev.txt")
  with open(result_file, 'w') as fout:
    for job in finished_jobs:
      if job.is_succeed:
        fout.write("job_{}, dev_acc:{:<.5f}, {}\n".format(job.job_id, job.result, job.settings))
        all_result_log = ""
        for key in sorted(job.all_result.keys()):
          all_result_log += " {}={:<.5f}".format(key, job.all_result[key])
        all_result_log += "\n"
        fout.write(all_result_log)
      else:
        fout.write("{}\t{}\t{}\n".format(job.job_id, job.ex_info, job.settings))
