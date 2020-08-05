#!/usr/bin/env python

import argparse
import sys
import os
from sweeper import Sweeper
from sweep_params import sweep_params

def check_sim_dir(sim_dir):
  files = os.listdir(sim_dir)
  for filename in ["env.txt", "gem5.cfg", "model_files", "run.sh",
                   "smv-accel.cfg", "trace.sh"]:
    if filename not in files:
      print("Could not find %s in this simulation directory!" % filename)
      sys.exit(1)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("sim_dir")
  args = parser.parse_args()

  check_sim_dir(args.sim_dir)

  sweeper = Sweeper(args.sim_dir, sweep_params)

  # Start enumerating all the configurations.
  sweeper.enumerate()

  # Start running simulations for all the generated configurations.
  # Use 20 threads to run the simulations in parallel.
  sweeper.runAll(16)

if __name__ == "__main__":
  main()
