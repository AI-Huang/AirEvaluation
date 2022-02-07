#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-30-20 16:37
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

"""GPU utils
"""
import subprocess as sp


def get_gpu_memory():
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])
                          for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values


def get_available_gpu_indices(gpus_memory, required_memory=10240):
    if_enough_memory = [_ > required_memory for _ in gpus_memory]
    available_gpu_indices = [
        i for i, enough in enumerate(if_enough_memory) if enough]
    assert len(available_gpu_indices) >= 2

    return available_gpu_indices


def main():
    pass


if __name__ == "__main__":
    main()
