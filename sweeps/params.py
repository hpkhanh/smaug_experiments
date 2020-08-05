import os
from configparser import RawConfigParser
import fileinput

def change_config_file(point_dir, config_file, kv_map):
  f = fileinput.input(os.path.join(point_dir, config_file), inplace=True)
  for line in f:
    for k in kv_map:
      if k in line:
        line = line % kv_map
    print(line, end="")
  f.close()

class BaseParam:
  def __init__(self, sweep_vals, changes_trace=False):
    self._sweep_vals = sweep_vals
    self._changes_trace = changes_trace
    self._curr_sweep_idx = -1

  def __str__(self):
    return "%s:%s" % (self._name, str(self.curr_sweep_value()))

  @property
  def changes_trace(self):
    return self._changes_trace

  def curr_sweep_value(self):
    return self._sweep_vals[self._curr_sweep_idx]

  def apply(self, point_dir):
    raise NotImplementedError

  def next(self):
    self._curr_sweep_idx += 1
    if self._curr_sweep_idx == len(self._sweep_vals):
      self._curr_sweep_idx = -1
      return False
    return True

class NumThreadsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "Number of threads"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {
            "num-threads": self.curr_sweep_value(),
            "num-cpus": self.curr_sweep_value()
        })

class NumAccelsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, True)
    self._name = "Number of accelerators"

  def apply(self, point_dir):
    # Change the run.sh, gem5.cfg and trace.sh.
    change_config_file(
        point_dir, "run.sh", {"num-accels": self.curr_sweep_value()})
    if self._sweep_vals[self._curr_sweep_idx] > 1:
      self._change_gem5_cfg(sweeper)
    change_config_file(
        point_dir, "trace.sh", {"num-accels": self.curr_sweep_value()})

  def _change_gem5_cfg(self, sweeper):
    gem5cfg = RawConfigParser()
    gem5cfg.read(os.path.join(sweeper.curr_point_dir(), "gem5.cfg"))
    acc0 = gem5cfg.sections()[0]
    acc0_id = int(gem5cfg.get(acc0, "accelerator_id"))
    with open(os.path.join(sweeper.curr_point_dir(), "gem5.cfg"), "w") as cfg:
      for n in range(1, self.curr_sweep_value()):
        new_acc = "acc" + str(n)
        gem5cfg.add_section(new_acc)
        for key, value in gem5cfg.items(acc0):
          gem5cfg.set(new_acc, key, value)
          gem5cfg.set(new_acc, "accelerator_id", str(acc0_id + n))
          gem5cfg.set(
              new_acc, "trace_file_name", "./dynamic_trace_acc%d.gz" % n)
      gem5cfg.write(cfg)

class SoCInterfaceParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "SoC interface"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "model_files", {"soc_interface": self.curr_sweep_value()})

class L2SizeParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "L2 size"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l2_size": self.curr_sweep_value()})

class L2AssocParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "L2 assoc"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l2_assoc": self.curr_sweep_value()})

class AccClockParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "Accelerator clock time"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh",
        {"sys-clock": "%.3fGHz" % (1.0 / self.curr_sweep_value())})
    change_config_file(
        point_dir, "smv-accel.cfg", {"cycle_time": self.curr_sweep_value()})
    change_config_file(
        point_dir, "gem5.cfg", {"cycle_time": self.curr_sweep_value()})

class MemTypeParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self._name = "Memory type"

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"mem-type": self.curr_sweep_value()})
