import os
import sys
import errno
import six
import shutil
import subprocess
import multiprocessing as mp
from params import *

param_types = {
    "num_threads": NumThreadsParam,
    "num_accels": NumAccelsParam,
    "soc_interface": SoCInterfaceParam,
    "l2_size": L2SizeParam,
    "l2_assoc": L2AssocParam,
    "acc_clock": AccClockParam
}

class Sweeper:
  def __init__(self, sim_dir, params):
    self._sim_dir = os.path.abspath(sim_dir)
    self._init_params(params)
    self._num_data_points = 0
    self._traces = set()
    # Create a folder for storing all the traces.
    self._trace_dir = os.path.join(self._sim_dir, "traces")
    if not os.path.isdir(self._trace_dir):
      os.mkdir(self._trace_dir)

  def _init_params(self, params):
    self._params = []
    for p, v in params.items():
      self._params.append(param_types[p](v))

  def curr_config_dir(self):
    return os.path.join(self._sim_dir, str(self._num_data_points))

  def _create_point(self):
    print("---Create data point: %d.---" % self._num_data_points)
    if not os.path.isdir(self.curr_config_dir()):
      os.mkdir(self.curr_config_dir())
    src_dir = self._sim_dir
    dst_dir = self.curr_config_dir()
    for f in ["gem5.cfg", "run.sh", "model_files", "trace.sh", "smv-accel.cfg"]:
      shutil.copyfile(
          os.path.join(self._sim_dir, f), os.path.join(
              self.curr_config_dir(), f))
    if not os.path.exists(os.path.join(self.curr_config_dir(), "env.txt")):
      os.symlink(
          os.path.join(self._sim_dir, "env.txt"),
          os.path.join(self.curr_config_dir(), "env.txt"))
    for param in self._params:
      param.apply(self)

    # Now all the configuration files have been updated, Check if we need to
    # generate new trace for this data point.
    trace_id = ""
    num_accels = 0
    for param in self._params:
      if param.changesTrace == True:
        if trace_id == "":
          trace_id = str(param)
        else:
          trace_id += "_" + str(param)
      if param.name == "num_accels":
        num_accels = param.sweep_vals[param.curr_sweep_idx]
    # Before we generate any traces, create links to the traces.
    for i in range(num_accels):
      link = os.path.join(self.curr_config_dir(), "dynamic_trace_acc%d.gz" % i)
      target = os.path.join(
          self._trace_dir, trace_id, "dynamic_trace_acc%d.gz" % i)
      try:
        os.symlink(target, link)
      except OSError as e:
        if e.errno == errno.EEXIST:
          os.remove(link)
          os.symlink(target, link)
        else:
          raise e
    # If this is a new trace id, generate new traces.
    if trace_id not in self._traces:
      self._traces.add(trace_id)
      trace_dir = os.path.join(self._trace_dir, trace_id)
      if not os.path.isdir(trace_dir):
        os.mkdir(trace_dir)
      # Run trace.sh to generate the traces.
      process = subprocess.Popen(["bash", "trace.sh"],
                                 cwd=self.curr_config_dir(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = process.communicate()
      assert process.returncode == 0, (
          "Generating trace returned nonzero exit code! Contents of output:\n "
          "%s\n%s" % (six.ensure_text(stdout), six.ensure_text(stderr)))

  def enumerate(self, param_idx=0):
    """Create configurations for all data points.  """
    if param_idx < len(self._params) - 1:
      while self._params[param_idx].next(self) == True:
        self.enumerate(param_idx + 1)
      return
    else:
      while self._params[param_idx].next(self) == True:
        self._create_point()
        self._num_data_points += 1
      return

  def runAll(self, threads):
    """Run simulations for all data points.

    Args:
      Number of threads used to run the simulations.
    """
    print("Running all data points.")
    counter = mp.Value('i', 0)
    pool = mp.Pool(
        initializer=_init_counter, initargs=(counter, ), processes=threads)
    for p in range(self._num_data_points):
      cmd = os.path.join(self._sim_dir, str(p), "run.sh")
      pool.apply_async(_run_simulation, args=(cmd, ))
    pool.close()
    pool.join()

finished_points = 0

def _init_counter(args):
  global counter
  counter = args

def _run_simulation(cmd):
  global counter
  process = subprocess.Popen(["bash", cmd], cwd=os.path.dirname(cmd),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  assert process.returncode == 0, (
      "Running simulation returned nonzero exit code! Contents of output:\n "
      "%s\n%s" % (six.ensure_text(stdout), six.ensure_text(stderr)))
  with counter.get_lock():
    counter.value += 1
  print("Finished points: %d" % counter.value)
