import os
from configparser import ConfigParser
import fileinput

class BaseParam:
  def __init__(self, sweep_vals, changesTrace=False):
    self.sweep_vals = sweep_vals
    self.changesTrace = changesTrace
    self.curr_sweep_idx = -1

  def __str__(self):
    return self.name + "_" + str(self.sweep_vals[self.curr_sweep_idx])

  def apply(self, sweeper):
    raise NotImplementedError

  def next(self, sweeper):
    self.curr_sweep_idx += 1
    if self.curr_sweep_idx == len(self.sweep_vals):
      self.curr_sweep_idx = -1
      return False
    return True

  def change_config_file(self, config_file, origin, new, sweeper):
    f = fileinput.input(
        os.path.join(sweeper.curr_config_dir(), config_file), inplace=True)
    for line in f:
      print(line.replace(origin, new), end="")
    f.close()

class NumThreadsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "num_threads"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--num-threads=1",
        "--num-threads=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)
    self.change_config_file(
        "run.sh", "--num-cpus=1",
        "--num-cpus=" + str(self.sweep_vals[self.curr_sweep_idx] + 1), sweeper)

class NumAccelsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, True)
    self.name = "num_accels"

  def apply(self, sweeper):
    #print self
    # Change the run.sh, gem5.cfg and trace.sh.
    self.change_config_file(
        "run.sh", "--num-accels=1",
        "--num-accels=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)
    if self.sweep_vals[self.curr_sweep_idx] > 1:
      self.change_gem5_cfg(sweeper)
    self.change_config_file(
        "trace.sh", "--num-accels=1",
        "--num-accels=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

  def change_gem5_cfg(self, sweeper):
    gem5cfg = ConfigParser()
    gem5cfg.read(os.path.join(sweeper.curr_config_dir(), "gem5.cfg"))
    acc0 = gem5cfg.sections()[0]
    acc0_id = int(gem5cfg.get(acc0, "accelerator_id"))
    with open(os.path.join(sweeper.curr_config_dir(), "gem5.cfg"), "w") as cfg:
      for n in range(1, self.sweep_vals[self.curr_sweep_idx]):
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
    self.name = "soc_interface"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "model_files", "smv_dma_topo.pbtxt",
        "smv_%s_topo.pbtxt" % self.sweep_vals[self.curr_sweep_idx], sweeper)

class L2SizeParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "l2_size"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--l2_size=2097152",
        "--l2_size=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

class L2AssocParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "l2_assoc"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--l2_assoc=16",
        "--l2_assoc=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

class AccClockParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "accel_clock_time"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--sys-clock=1GHz",
        "--sys-clock=%.3fGHz" % (1.0 / self.sweep_vals[self.curr_sweep_idx]),
        sweeper)
    self.change_config_file(
        "smv-accel.cfg", "cycle_time,1",
        "cycle_time,%d" % self.sweep_vals[self.curr_sweep_idx], sweeper)
    self.change_config_file(
        "gem5.cfg", "cycle_time = 1",
        "cycle_time = %d" % self.sweep_vals[self.curr_sweep_idx], sweeper)
