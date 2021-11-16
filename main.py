import time
import os

import cv2
import imageio
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

from deep_sort_tools import generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


if __name__ == "__main__":
    
    # Input - Output Path
    input_path = r"dataset/pexels-kelly-lacy-5473765.mp4"
    output_path = os.path.join("output", "testing")
    os.makedirs(output_path, exist_ok=True)
    # Experiment Setup, Loading Model
    exp = get_exp(r"yolox-exps/default/yolox_x.py", None)
    exp.test_conf = 0.5
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)
    model = exp.get_model()
    model.cuda()
    model.eval()
    # Load Weight from check point
    ckpt_file = r"models/yolox_x.pth"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    # Predictor
    predictor = Predictor(model, exp, COCO_CLASSES, None, "gpu", False, False)
    current_time = time.localtime()

    ########################################################################################
    # Video Cap and details
    cap = cv2.VideoCapture(input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Save folder setup
    save_folder = os.path.join(output_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, input_path.split("/")[-1]) # File name
    # Writer Obj
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    
    # Deep_Sort Tracker
    model_filename = r"deep_sort_network/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    max_cosine_distance = 0.3
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=5)

    # Main Loop
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            imSize = frame.shape
            outputs, img_info = predictor.inference(frame)
            outputs = outputs[0].cpu().detach().numpy()
            outputs[:,:4] *= imSize[1]/exp.test_size[0]
            bboxes = outputs.copy()
            bboxes[:,2] = outputs[:,2]-outputs[:,0]
            bboxes[:,3] = outputs[:,3]-outputs[:,1]
            features = encoder(frame, bboxes)
            detections = [Detection(bbox[:4], bbox[5], feature, bbox[6]) for bbox, feature in zip(bboxes, features)]
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                t_bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(t_bbox[0]), int(t_bbox[1])), (int(t_bbox[2]), int(t_bbox[3])),(255,0,0),2)
                cv2.putText(frame, COCO_CLASSES[int(track.classID)] + " id: " + str(track.track_id), (int(t_bbox[0]), int(t_bbox[1])-10),
                            0, 0.60, (255,255,255),1)
            for i  in range(outputs.shape[0]):
                cv2.rectangle(frame, (int(outputs[i, 0]), int(outputs[i, 1])), (int(outputs[i, 2]), int(outputs[i, 3])),(0,0,255),1)
                # cv2.putText(frame, COCO_CLASSES[int(outputs[i,6])],(int(outputs[i, 0]), int(outputs[i, 1]-10)),0, 0.50, (255,255,255),1)
            cv2.imshow("Output Video", frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
cv2.destroyAllWindows()