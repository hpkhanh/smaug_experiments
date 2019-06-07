#!/usr/bin/env bash

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../nnet_lib/build
model_dir=`git rev-parse --show-toplevel`/models

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --env=env.txt \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=DDR3_1600_8x8  \
  --sys-clock=1GHz \
  --cpu-type=TimingSimpleCPU \
  --caches \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  -c ${bmk_dir}/smaug \
  -o "${model_dir}/minerva/minerva_smv_topo.pbtxt ${model_dir}/minerva/minerva_smv_params.pb --gem5 --debug-level=0" \
  | tee stdout
  # | gzip -c > stdout.gz
