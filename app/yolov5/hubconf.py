# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch
from pathlib import WindowsPath as Path  # ✅ Force WindowsPath for compatibility

from models.common import AutoShape, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
from utils.downloads import attempt_download
from utils.general import LOGGER, check_requirements, intersect_dicts, logging
from utils.torch_utils import select_device


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(str(name))  # ✅ Force conversion to WindowsPath
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name

    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('⚠️ ClassificationModel is not AutoShape compatible.')
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('⚠️ SegmentationModel is not AutoShape compatible.')
                    else:
                        model = AutoShape(model)
            except Exception:
                model = attempt_load(path, device=device, fuse=False)
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]
            model = DetectionModel(cfg, channels, classes)
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])
                model.load_state_dict(csd, strict=False)
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names
        if not verbose:
            LOGGER.setLevel(logging.INFO)
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


# Exposed functions
def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5n', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5m', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5l', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5x', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5n6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5s6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5m6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5l6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create('yolov5x6', pretrained, channels, classes, autoshape, _verbose, device)
