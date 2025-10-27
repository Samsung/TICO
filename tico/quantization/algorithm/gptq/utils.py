# Copyright (c) 2024 Intel Corporation
# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def find_layers(module, layers=[torch.nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def gather_single_batch_from_dict(data_dict, idx):
    """
    Gather single batch from a dict.

    Args:
        data_dict (dict): data dict.
        idx (int): index

    Returns:
        dict: single batch.
    """
    # obtain a set of keyword input from cache
    single_batch = {}
    for k, v in data_dict.items():
        single_batch[k] = data_dict[k][idx]
    return single_batch


def gather_single_batch_from_list(data_list, idx):
    """
    Gather single batch from a list.

    Args:
        data_dict (dict): data list.
        idx (int): index

    Returns:
        list: single batch.
    """
    # obtain a set of positional input from cache
    single_batch = []
    for data_item in data_list:
        single_batch.append(data_item[idx])
    return single_batch
