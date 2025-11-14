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

import argparse
import contextlib
import io
import os
import time
from enum import Enum

import torch
import torch.distributed as dist
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.fpi_gptq import FPIGPTQConfig
from tico.quantization.config.rtn import RTNConfig

list_of_available_models = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_5",
    "swin_v2_b",
    "swin_v2_t",
]


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def prepare_data_loader(dir):
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = datasets.ImageFolder(
        dir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True
    )

    return loader


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # type: ignore[assignment]

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def validate(val_loader, model, criterion, num_of_images=5000):

    device = torch.device("cuda")
    for param in model.parameters():
        if hasattr(param, "device"):
            device = getattr(param, "device")
            break

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                if i > num_of_images:
                    break
                i = base_progress + i
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # if i % display_progress == 0:
                #    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg, top5.avg


def get_model(model_name: str):
    if model_name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
    elif model_name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(
            weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
    elif model_name == "squeezenet1_1":
        model = torchvision.models.squeezenet1_1(
            weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        )
    elif model_name == "shufflenet_v2_x1_5":
        model = torchvision.models.shufflenet_v2_x1_5(
            weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        )
    elif model_name == "convnext_large":
        model = torchvision.models.convnext_large(
            weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )
    elif model_name == "resnext101_64x4d":
        model = torchvision.models.resnext101_64x4d(
            weights=torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        )
    elif model_name == "vit_l_32":
        model = torchvision.models.vit_l_32(
            weights=torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1
        )
    elif model_name == "swin_v2_b":
        model = torchvision.models.swin_v2_b(
            weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1
        )
    elif model_name == "swin_v2_t":
        model = torchvision.models.swin_v2_t(
            weights=torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1
        )
    else:
        assert False

    model.eval()

    return model


def get_statistics_of_model_quantization(
    model,
    data_loader,
    weight_config,
    num_of_images_for_gptq_calibration=100,
    num_of_images_for_validation=5000,
    dev: str = "cuda",
):

    input_0, _ = next(iter(data_loader))

    model.eval()

    prepare(model, weight_config, args=(input_0.to(dev),), inplace=True)

    num_of_images_for_calibration = num_of_images_for_gptq_calibration
    for index, (image, _) in enumerate(data_loader):
        if index >= num_of_images_for_calibration:
            break
        with torch.no_grad():
            model(image.to(dev))

    tick = time.time()
    convert(model, inplace=True)
    q_time = time.time() - tick

    # Evaluate the quantized model
    acc1_quantized, acc5_quantized = validate(
        data_loader, model, criterion=torch.nn.CrossEntropyLoss().to(dev), num_of_images=num_of_images_for_validation
    )

    return q_time, acc1_quantized, acc5_quantized


def compare_FPI_quantization_of_model(
    model_name,
    data_loader,
    num_of_images_for_gptq_calibration,
    num_of_validation_images,
    dev: str = "cuda",
):
    with io.StringIO() as buffer, contextlib.redirect_stdout(
        buffer
    ), contextlib.redirect_stderr(buffer):
        model = get_model(model_name).to(dev)

        acc1_original, acc5_original = validate(
            data_loader,
            model,
            criterion=torch.nn.CrossEntropyLoss().to(dev),
            num_of_images=num_of_validation_images,
        )
        # no need to load_model as validate above does not change model
        model = model.to(dev)
        model = get_model(model_name).to(dev)
        (
            q_RTN_time,
            acc1_RTN_quantized,
            acc5_RTN_quantized,
        ) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=RTNConfig(),
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            num_of_images_for_validation=num_of_validation_images,
            dev=dev,
        )

        model = get_model(model_name).to(dev)
        (
            q_GPTQ_time,
            acc1_GPTQ_quantized,
            acc5_GPTQ_quantized,
        ) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=FPIGPTQConfig(quantize_convs_groupwise=False),#GPTQConfig(),
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            num_of_images_for_validation=num_of_validation_images,
            dev=dev,
        )

        model = get_model(model_name).to(dev)
        (
            q_GPTQGR_time,
            acc1_GPTQGR_quantized,
            acc5_GPTQGR_quantized,
        ) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=FPIGPTQConfig(quantize_convs_groupwise=True), #GPTQConfig(quantize_convs_groupwise=True)
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            dev=dev,
        )

    print(f"resuts_of_comparison_for {model_name}")
    print(f"acc1_original  {acc1_original:.4f} acc5_original  {acc5_original:.4f}")
    print(
        f"acc1_quantized {acc1_RTN_quantized:.4f} acc5_quantized {acc5_RTN_quantized:.4f} time_of_quantization {q_RTN_time:.4f} - RTN "
    )
    print(
        f"acc1_quantized {acc1_GPTQ_quantized:.4f} acc5_quantized {acc5_GPTQ_quantized:.4f} time_of_quantization {q_GPTQ_time:.4f} - GPTQ "
    )
    print(
        f"acc1_quantized {acc1_GPTQGR_quantized:.4f} acc5_quantized {acc5_GPTQGR_quantized:.4f} time_of_quantization {q_GPTQGR_time:.4f} - GPTQGR "
    )

    print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["all"],
        help="Possible models that can be used for comparison\n"
        "'all' - almost all of the below models\n"
        "'mobilenet_v2'\n"
        "'mobilenet_v3_small'\n"
        "'mobilenet_v3_large'\n"
        "'squeezenet1_1'\n"
        "'shufflenet_v2_x1_5'\n"
        "'convnext_large'\n"
        "'resnext101_64x4d'\n"
        "'vit_l_32'\n"
        "'swin_v2_b'\n"
        "'swin_v2_t'\n",
    )
    parser.add_argument(
        "--num_of_calibration_images",
        type=int,
        default=100,
        help="number of images to calibrate gptq",
    )
    parser.add_argument(
        "--num_of_validation_images",
        type=int,
        default=5000,
        help="number of images to vaidate result",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device for inference cuda/cpu"
    )

    args = parser.parse_args()
    models = []
    for name in args.models:
        assert name == "all" or name in list_of_available_models
        if name == "all":
            models.extend(list_of_available_models)
        else:
            models.append(name)

    data_loader = prepare_data_loader(dir="/mnt/storage/datasets/imagenet/val")
    for model_name in models:
        compare_FPI_quantization_of_model(
            model_name,
            data_loader,
            num_of_images_for_gptq_calibration=args.num_of_calibration_images,
            num_of_validation_images=args.num_of_validation_images,
            dev=args.device,
        )
