#!/usr/bin/env python

import argparse
import sys
import os
from sweeper import Sweeper
from sweep_params import sweep_params

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="Model name.")
  parser.add_argument(
      "--output-dir", help="Output directory for generating the data points.")
  parser.add_argument(
      "--run-points", action="store_true", default=False,
      help="Option to run the generated data points.")
  parser.add_argument(
      "--num-threads", type=int, default=8,
      help="Number of threads used to run the data points.")
  args = parser.parse_args()

  if not args.model:
    raise ValueError("Please provide the model name!")
  if not args.output_dir:
    raise ValueError("Please provide the output directory!")
  sweeper = Sweeper(args.model, args.output_dir, sweep_params)

  # Start enumerating all the data points.
  sweeper.enumerate_all()

  # Start running simulations for all the generated data points.
  if args.run_points:
    sweeper.run_all(threads=args.num_threads)

if __name__ == "__main__":
  main()
