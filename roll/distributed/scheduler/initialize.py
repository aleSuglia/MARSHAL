import os
import subprocess
import sys
import time

import ray

from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()

default_envs = {
    # "RAY_DEBUG": "legacy"
    "TORCHINDUCTOR_COMPILE_THREADS": "2",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_CUMEM_ENABLE": "0",   # https://github.com/NVIDIA/nccl/issues/1234
    "NCCL_NVLS_ENABLE": "0",
    "ACCL_TUNING_LEVEL": "1",
}
if os.environ.get("LD_LIBRARY_PATH"):
    # Force-propagate the driver's own LD_LIBRARY_PATH (e.g. set by `module load cuda/...`
    # on HPC systems) into every Ray actor's runtime_env, the same way the CUDA/NCCL vars
    # above already are. Without this, an actor's own process can have working CUDA while
    # a subprocess it spawns internally (e.g. sglang's scheduler, via mp.Process in
    # roll/third_party/sglang/v046post4_patch/engine.py) fails at exec time with
    # "error while loading shared libraries: libcudart.so.12: cannot open shared object
    # file" -- seen on Eddie, where libcudart is only resolvable via the module-set
    # LD_LIBRARY_PATH, not bundled in the pip-installed torch wheel's own rpath.
    default_envs["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        # cmd = f"ray start --head --port={master_port} --node-name={node_name}"
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 0
        cmd = f"ray start --head --port={master_port} --node-name={node_name} --num-gpus={num_gpus}"
    else:
        # fix: 处理大规模下可能会出现的head/worker node创建顺序不一致问题
        time.sleep(5)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def init():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    manual_start = start_ray_cluster()
    runtime_env = {
        "env_vars": default_envs
    }

    if not ray.is_initialized():
        ray.init(
            address=f"{master_addr}:{master_port}" if manual_start else None,
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=not manual_start,
            runtime_env=runtime_env,
        )
        logger.info("Ray cluster initialized")

    if manual_start:
        wait_for_nodes(expected=world_size)
        listener = LogMonitorListener()
        listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")

    if manual_start and rank > 0:
        sys.exit(0)
