#!/usr/bin/env bash

source ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --env=env.txt \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=%(mem-type)s  \
  --sys-clock=%(sys-clock)s \
  --cpu-clock=%(cpu-clock)s \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l2_size=%(l2_size)s \
  --l2_assoc=%(l2_assoc)s \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${SMAUG_HOME}/build/bin/smaug \
  -o "${topo_file} ${params_file} --sample-level=high --gem5 --debug-level=0 --num-accels=%(num-accels)s" \
  > stdout 2> stderr
