import argparse
import contextlib
import copy
import datetime
import io
import os
import time
from collections import defaultdict, deque
from contextlib import redirect_stdout

import numpy as np

# from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util
import test_tconv_transforms as T
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tico.quantization import convert, prepare
from tico.quantization.config.fpi_gptq import FPIGPTQConfig
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.rtn import RTNConfig

# from torchmetrics.detection import MeanAveragePrecision

# from torchvision.io.image import decode_image
# from torchvision.models.detection import *
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.models.resnet import ResNet50_Weights

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

list_of_available_models = [
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "keypointrcnn_resnet50_legacy",
    # "keypointrcnn_resnet50_fpn",
]


def get_torch_model_name_and_weights(model_name: str):
    torch_model_name = None
    weights_name = None
    if model_name == "maskrcnn_resnet50_fpn":
        torch_model_name = "maskrcnn_resnet50_fpn"
        weights_name = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    elif model_name == "maskrcnn_resnet50_fpn_v2":
        torch_model_name = "maskrcnn_resnet50_fpn_v2"
        weights_name = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    elif model_name == "keypointrcnn_resnet50_legacy":
        torch_model_name = "keypointrcnn_resnet50_fpn"
        weights_name = KeypointRCNN_ResNet50_FPN_Weights.COCO_LEGACY
    elif model_name == "keypointrcnn_resnet50_fpn":
        torch_model_name = "keypointrcnn_resnet50_fpn"
        weights_name = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    else:
        assert False

    return torch_model_name, weights_name


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = mask_util.frPyObjects(polygons, height, width)
        mask = mask_util.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": [], "info": {}}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for mask_util
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = mask_util.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(
    root,
    image_set,
    transforms,
    mode="instances",
    use_v2=False,
    with_masks=False,
    num_samples=-1,
):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": (
            "train2017",
            os.path.join("annotations", anno_file_template.format(mode, "train")),
        ),
        "val": (
            "val2017",
            os.path.join("annotations", anno_file_template.format(mode, "val")),
        ),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(
            img_folder, ann_file, transforms=transforms
        )
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # TODO: handle with_masks for V1?
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if num_samples > 0 and num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, [i for i in range(num_samples)])

    return dataset


def get_dataset(args, model, model_weights, num_samples=-1):
    image_set = "val"
    dataset = "coco_kp" if "keypoint" in model else "coco"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[
        dataset
    ]
    with_masks = "mask" in model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(model_weights),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
        num_samples=num_samples,
    )
    return ds, num_classes


def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        import test_tconv_transforms as reference_transforms

        return reference_transforms, None


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        else:
            raise ValueError(
                f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}"
            )

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(weights):
    weights = torchvision.models.get_weight(weights)
    trans = weights.transforms()
    return lambda img, target: (trans(img), target)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    return [data]


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = 1
    if world_size < 2:
        return input_dict
    # with torch.inference_mode():
    #    names = []
    #    values = []
    #    # sort the keys so that they are consistent across processes
    #    for k in sorted(input_dict.keys()):
    #        names.append(k)
    #        values.append(input_dict[k])
    #    values = torch.stack(values, dim=0)
    #    dist.all_reduce(values)
    #    if average:
    #        values /= world_size
    #    reduced_dict = {k: v for k, v in zip(names, values)}
    # return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(
                f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}"
            )
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def evaluate(self, imgs):
        with redirect_stdout(io.StringIO()):
            imgs.evaluate()
        return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(
            -1, len(imgs.params.areaRng), len(imgs.params.imgIds)
        )

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = self.evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.to(device)
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def collate_fn(batch):
    return tuple(zip(*batch))


def validate(data_loader, model, dev):
    # evaluate
    prev_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    evaluator = evaluate(model, data_loader, device=dev)
    accuracy = {}
    for key in evaluator.coco_eval:
        accuracy[key] = evaluator.coco_eval[key].stats[0]
        print(f"{key} accuracy - {evaluator.coco_eval[key].stats[0]:.3f}")

    torch.backends.cudnn.deterministic = prev_deterministic
    return accuracy


def get_statistics_of_model_quantization(
    model,
    data_loader,
    weight_config,
    num_of_images_for_gptq_calibration=100,
    dev: str = "cuda",
):
    model.eval()
    input_0, _ = next(iter(data_loader))
    input_0 = list(img.to(dev) for img in input_0)
    prepare(model, weight_config, args=input_0, inplace=True)

    for index, (images, _) in enumerate(data_loader):
        if index >= num_of_images_for_gptq_calibration:
            break
        with torch.no_grad():
            input = list(img.to(dev) for img in images)
            model(input)

    # quantize
    tick = time.time()
    convert(model, inplace=True)
    q_time = time.time() - tick
    print(f"time of quantization {q_time}")

    # evaluate
    accuracy = validate(data_loader, model, dev)

    return q_time, accuracy


def compare_FPI_quantization_of_model(
    args,
    model_name,
    num_of_images_for_gptq_calibration,
    num_of_validation_images,
    dev: str = "cuda",
):
    with io.StringIO() as buffer, contextlib.redirect_stdout(
        buffer
    ), contextlib.redirect_stderr(buffer):
        torch_model_name, weights_name = get_torch_model_name_and_weights(model_name)
        dataset_test, num_classes = get_dataset(
            args=args,
            model=torch_model_name,
            model_weights=str(weights_name),
            num_samples=num_of_validation_images,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            num_workers=4,
            collate_fn=collate_fn,
        )

        model = torchvision.models.get_model(
            torch_model_name,
            weights=str(weights_name),
            weights_backbone=None,
            num_classes=num_classes,
        )
        # is_conv3d = False
        # for name, module in model.named_modules():
        #    if isinstance(module, torch.nn.ConvTranspose2d) or isinstance(module, torch.nn.ConvTranspose1d) or isinstance(module, torch.nn.ConvTranspose3d):
        #        is_conv3d = True

        accuracy_original = validate(data_loader, model, dev)
        del model
        model = None
        # no need to load_model as validate above does not change model
        model = torchvision.models.get_model(
            torch_model_name,
            weights=str(weights_name),
            weights_backbone=None,
            num_classes=num_classes,
        ).to(dev)
        (q_RTN_time, accuracy_RTN_quantized,) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=RTNConfig(),
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            dev=dev,
        )
        del model
        model = None
        model = torchvision.models.get_model(
            torch_model_name,
            weights=str(weights_name),
            weights_backbone=None,
            num_classes=num_classes,
        ).to(dev)
        (q_GPTQ_time, accuracy_GPTQ_quantized) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=GPTQConfig(),
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            dev=dev,
        )

        model = torchvision.models.get_model(
            torch_model_name,
            weights=str(weights_name),
            weights_backbone=None,
            num_classes=num_classes,
        ).to(dev)
        (
            q_FPI_GPTQ_time,
            accuracy_FPI_GPTQ_quantized,
        ) = get_statistics_of_model_quantization(
            model,
            data_loader,
            weight_config=FPIGPTQConfig(),
            num_of_images_for_gptq_calibration=num_of_images_for_gptq_calibration,
            dev=dev,
        )
        del model
        model = None

    print(f"resuts_of_comparison_for {model_name}")
    acc_original = ", ".join(
        f"{key}: {value:.3f}" for key, value in accuracy_original.items()
    )
    print(acc_original)
    acc_RTN = ", ".join(
        f"{key}: {value:.3f}" for key, value in accuracy_RTN_quantized.items()
    )
    print(f"{acc_RTN} time_of_quantization {q_RTN_time:.4f} - RTN")
    acc_FPI_GPTQ = ", ".join(
        f"{key}: {value:.3f}" for key, value in accuracy_FPI_GPTQ_quantized.items()
    )
    print(f"{acc_FPI_GPTQ} time_of_quantization {q_FPI_GPTQ_time:.4f} - FPI_GPTQ")
    acc_GPTQ = ", ".join(
        f"{key}: {value:.3f}" for key, value in accuracy_GPTQ_quantized.items()
    )
    print(f"{acc_GPTQ} time_of_quantization {q_GPTQ_time:.4f} - GPTQ")
    print(
        f"{100 * (1.0 - q_FPI_GPTQ_time / q_GPTQ_time):.2f}% speed-up for {model_name}"
    )

    print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--data-path",
        default="/mnt/storage/datasets/ms_coco",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["all"],
        help="Possible models that can be used for comparison\n"
        "'all' - almost all of the below models\n"
        "'maskrcnn_resnet50_fpn'\n"
        "'maskrcnn_resnet50_fpn_v2'\n"
        "'keypointrcnn_resnet50_legacy'\n"
        "'keypointrcnn_resnet50_fpn'\n",
    )

    # parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    # parser.add_argument("--model", default="maskrcnn_resnet50_fpn_v2", type=str, help="model name")
    # parser.add_argument("--model", default="keypointrcnn_resnet50_fpn", type=str, help="model name")

    # parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    # parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")
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
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument(
        "--rpn-score-thresh",
        default=None,
        type=float,
        help="rpn score threshold for faster-rcnn",
    )

    args = parser.parse_args()
    print(args)
    model_names = []
    for name in args.models:
        assert name == "all" or name in list_of_available_models
        if name == "all":
            model_names.extend(list_of_available_models)
        else:
            model_names.append(name)

    for model_name in model_names:
        compare_FPI_quantization_of_model(
            args,
            model_name,
            args.num_of_calibration_images,
            args.num_of_validation_images,
            dev=args.device,
        )
