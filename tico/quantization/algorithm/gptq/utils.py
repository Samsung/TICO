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
import tqdm


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


def get_dataset_for_calibration(model, dataset):
    """Enrich dataset with model ouputs to be used as targets"""

    class DataSetWithLabels(torch.utils.data.Dataset):
        def __init__(self, inputs, targets, transform=None):
            self.n_inputs = len(inputs)
            self.inputs = inputs
            self.labels = targets
            self.transform = transform

        def __len__(self):
            return self.n_inputs

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sample = self.inputs[idx]
            if self.transform:
                sample = self.transform(sample)

            return (sample, self.labels[idx])

    targets = []

    with torch.no_grad():
        print("Computing calibration set")
        for prompt in tqdm.tqdm(dataset):
            results = model(prompt.to(model.device)).logits.detach()
            results = torch.argmax(results.detach(), dim=-1).cpu()

            targets.append(results)

    labeled_data = DataSetWithLabels(dataset, targets)
    dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=1, shuffle=False)
    return dataloader


class SensitivityCalibrator:
    """
    Sensitivity calibrator - compute sensitivies using empirical Fisher information.

    Sensitivities are assumed to mimick second order derivatives.
    They can be used to estimate `logits` global error introduced by quantization.
    So here we use empirical Fisher information to assess diagonal second order derivatives.
    Please see https://arxiv.org/abs/1905.12558?ref=inference.vc for a discussion.
    """

    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def compute_sensitivity_info(self):

        data_loader = get_dataset_for_calibration(self.model, self.dataset)

        dtype = self.model.dtype
        model = self.model.float()

        sensitivity = {}
        modules_to_process = {}
        name_of_module: dict[torch.nn.Linear, str] = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_to_process[name] = module
                name_of_module[module] = name
                sensitivity[name] = torch.zeros_like(module.weight).cpu()

        print("Calibrating sensitivity")
        for inputs, targets in tqdm.tqdm(data_loader):
            model.zero_grad()
            inp_ids = inputs.view(-1, inputs.shape[-1])
            logits = model(inp_ids.to(model.device)).logits

            outputs = logits.squeeze()
            targets = targets.squeeze()

            t_index = outputs.shape[0] - 1  # priority to the last token
            outputs_el = outputs[t_index : t_index + 1, :]  # noqa E203
            targets_el = targets[t_index : t_index + 1]  # noqa E203

            model.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(
                outputs_el, targets_el.to(model.device)
            )  # for Fisher this must be CrossEntropy

            loss.backward(retain_graph=False)

            # update second order information as current weights gradients are ready
            for name in modules_to_process:
                cur_module = modules_to_process[name]
                cur_grad = cur_module.weight.grad.detach().clone()
                if torch.isnan(cur_grad).any().item():
                    print("WARNING NaN detected")

                sensitivity[name] += torch.mul(cur_grad, cur_grad).cpu()

                cur_grad = None
                del cur_grad

                if model.device.type != "cpu":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            loss.detach()

            logits = outputs = targets = loss = None
            del loss, logits, outputs, targets

            if model.device.type != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        for name in modules_to_process:
            sensitivity[name] /= len(data_loader)

        model = model.to(dtype)

        return sensitivity
