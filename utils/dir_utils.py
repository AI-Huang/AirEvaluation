#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-06-20 14:33
# @Update  : Nov-25-20 03:15
# @Update  : Nov-27-20 16:32
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

import os
import errno
import six
import shutil

# in tensorflow.python.keras.utils import data_utils


def makedir_exist_ok(dirpath):
    """makedir_exist_ok compatible for both Python 2 and Python 3
    """
    if six.PY3:
        os.makedirs(
            dirpath, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
    else:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_exper_suffixes(prefix):
    exper_suffixes = []
    log_dir = os.path.join(prefix, "logs")
    for root, dirs, files in os.walk(log_dir):
        # if the "train" directory is found
        if "train" in dirs:
            exper_suffixes.append(os.path.relpath(root, log_dir))
    return exper_suffixes


def delete_emtpy_directory(prefix="."):
    for root, dirs, files in os.walk(prefix):
        if not os.listdir(root):
            print(f"removing: {root}")
            os.rmdir(root)


def delete_experiment_data(suffix, prefix="."):
    """delete_experiment_data
    Delete experiment data according to its suffix (model_type and exper_id).
    Inputs:
        suffix: *this* experiment's suffix path.
        prefix: experiment data root path, where ckpts and logs directories should exist.
    """
    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts"))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs"))

    dir_to_remove = os.path.join(ckpt_dir, suffix)
    if os.path.isdir(dir_to_remove):
        print(f"removing: {dir_to_remove}")
        shutil.rmtree(dir_to_remove, ignore_errors=True)

    dir_to_remove = os.path.join(log_dir, suffix)
    if os.path.isdir(dir_to_remove):
        print(f"removing: {dir_to_remove}")
        shutil.rmtree(dir_to_remove, ignore_errors=True)
