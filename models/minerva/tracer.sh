#!/usr/bin/env bash
model_dir=`git rev-parse --show-toplevel`/models
topo_file=${model_dir}/minerva/minerva_smv_topo.pbtxt
params_file=${model_dir}/minerva/minerva_smv_params.pb

/workspace/smaug/build/bin/smaug-instrumented ${topo_file} ${params_file} --debug-level=1 --sample-level=no --num-threads --num-accels=1 --network-config=/workspace/smaug/smaug/layers_minerva.cfg
# /workspace/smaug/build/bin/smaug ${topo_file} ${params_file} --sample-level=no --gem5 --debug-level=1 --num-threads --num-accels=1 --network-config=/workspace/smaug/smaug/layers_minerva.cfg