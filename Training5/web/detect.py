import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, select_device, time_synchronized

# Load model
model = attempt_load("weights/last.pt", map_location="cpu")  # load FP32 model


def detect(config, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = (
        config["output"],
        config["source"],
        config["weights"],
        config["view_img"],
        config["save_txt"],
        config["img_size"],
    )
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://"))
    )

    # Initialize
    set_logging()
    device = select_device(config["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=config["augment"])[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            config["conf_thres"],
            config["iou_thres"],
            classes=config["classes"],
            agnostic=config["agnostic_nms"],
        )
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            save_path = str(Path(out) / Path(p).name)  # img.jpg
            txt_path = str(Path(out) / Path(p).stem) + (
                "_%g" % dataset.frame if dataset.mode == "video" else ""
            )
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or view_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=3,
                        )

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        print("Results saved to %s" % Path(out))
        # if platform.system() == "Darwin" and not config["update"]:  # MacOS
        #     os.system("open " + save_path)

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    config = {
        "weights": "yolov5s.pt",
        "source": "inference/images",
        "output": "inference/output",
        "img_size": 640,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "device": "cpu",
        "view_img": False,
        "save_txt": "store_true",
        "classes": "",
        "agnostic_nms": "store_true",
        "augment": "store_true",
        "update": False,
    }

    with torch.no_grad():
        if config["update"]:  # update all models (to fix SourceChangeWarning)
            for config["weights"] in [
                "yolov5s.pt",
                "yolov5m.pt",
                "yolov5l.pt",
                "yolov5x.pt",
            ]:
                detect(config)
                strip_optimizer(config["weights"])
        else:
            detect(config)
