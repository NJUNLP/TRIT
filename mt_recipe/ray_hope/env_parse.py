import json
import os
import socket


def get_afo_env_spec():
    env_var = "WRITE_SPEC_ENV_TO_FILE"
    cluster_spec = None
    if env_var in os.environ:
        try:
            # 读取JSON文件
            FILE_PATH = "/workdir/AFO_ENV_CLUSTER_SPEC.env"
            with open(FILE_PATH, encoding="utf-8") as f:
                cluster_spec = json.load(f)
        except FileNotFoundError:
            print(f"文件未找到：{FILE_PATH}")
        except json.JSONDecodeError:
            print(f"文件内容不是合法的JSON格式：{FILE_PATH}")
    else:
        if "AFO_ENV_CLUSTER_SPEC" in os.environ:
            cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    return cluster_spec


def get_master_addr():
    afo_env_spec = get_afo_env_spec()
    master_info = afo_env_spec["worker"][0]
    master_addr, master_ports = master_info.split(":")
    master_ip = socket.gethostbyname(master_addr)
    return master_ip


def get_node_rank():
    afo_env_spec = get_afo_env_spec()
    worker_index = afo_env_spec["index"]
    return worker_index


def get_nproc_per_node():
    nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
    return nproc_per_node


def get_nnodes():
    afo_env_spec = get_afo_env_spec()
    nnodes = len(afo_env_spec["worker"])
    return nnodes


def get_master_port():
    afo_env_spec = get_afo_env_spec()
    master_info = afo_env_spec["worker"][0]
    master_addr, master_ports = master_info.split(":")
    master_ports = master_ports.split(",")
    master_port = master_ports[0]
    return master_port
