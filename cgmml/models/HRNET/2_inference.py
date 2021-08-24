import logging
import os
import time

import cgmml.models.HRNET.code.models.pose_hrnet  # noqa
import cv2
import glob2 as glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from cgmml.models.HRNET.code.config import cfg, update_config
from cgmml.models.HRNET.code.config.constants import (COCO_KEYPOINT_INDEXES, NUM_KPTS)
from cgmml.models.HRNET.code.models.pose_hrnet import get_pose_net
from cgmml.models.HRNET.code.utils.utils import (box_to_center_scale, calculate_pose_score, draw_pose,
                                                 get_person_detection_boxes, get_pose_estimation_prediction)

logging.basicConfig(level=logging.INFO, filename='pose_prediction.log',
                    format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', filemode='w')


class PosePrediction:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_box_model(self):
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.box_model.to(self.ctx)
        self.box_model.eval()

    def load_pose_model(self):
        self.pose_model = get_pose_net(cfg)
        self.pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
        self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(self.ctx)
        self.pose_model.eval()
        self.pose_model

    def read_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)

    def preprocess_image(self):
        self.input = []
        self.img = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(
            self.img / 255.).permute(2, 0, 1).float().to(self.ctx)
        self.input.append(img_tensor)

    def orient_image_using_scan_type(self, scan_type):
        if scan_type in ['100', '101', '102']:
            self.rotated_image = cv2.rotate(self.image_bgr, cv2.ROTATE_90_CLOCKWISE)  # Standing
        elif scan_type in ['200', '201', '202']:
            self.rotated_image = cv2.rotate(self.image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Laying
        else:
            logging.info("%s %s %s", "Provided scan type", scan_type, "not supported")
            logging.info("Keeping the image in the same orientation as provided")
            self.rotated_image = self.image_bgr

    def perform_box_on_image(self):
        self.pred_boxes, self.pred_score = get_person_detection_boxes(
            self.box_model, self.input, threshold=cfg.BOX_MODEL.THRESHOLD)
        return self.pred_boxes, self.pred_score

    def perform_pose_on_image(self, idx):
        center, scale = box_to_center_scale(
            self.pred_boxes[idx], cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        self.pose_preds, self.pose_score = get_pose_estimation_prediction(
            self.pose_model, self.img, center, scale)
        return self.pose_preds, self.pose_score

    def pose_draw_on_image(self):
        if len(self.pose_preds) >= 1:
            for kpt in self.pose_preds:
                draw_pose(kpt, self.rotated_image)  # draw the poses

    def save_final_image(self, final_image_name):
        cv2.imwrite('outputs/' + final_image_name, self.rotated_image)


class ResultGeneration:
    def __init__(self, pose_prediction, save_pose_overlay):
        self.pose_prediction = pose_prediction
        self.save_pose_overlay = save_pose_overlay

    def result_on_artifact_level(self, jpg_path, scan_type):

        start_time = time.time()

        self.pose_prediction.read_image(jpg_path)
        self.pose_prediction.orient_image_using_scan_type(scan_type)
        self.pose_prediction.preprocess_image()

        pred_boxes, pred_score = self.pose_prediction.perform_box_on_image()

        pose_result = []

        for idx in range(len(pred_boxes)):
            single_body_pose_result = {}
            key_points_coordinate_list = []
            key_points_prob_list = []

            pose_bbox = pred_boxes[idx]
            pose_preds, pose_score = self.pose_prediction.perform_pose_on_image(idx)
            if self.save_pose_overlay:
                self.pose_prediction.pose_draw_on_image()
                # TODO Ensure save image path
                self.pose_prediction.save_final_image(jpg_path.split('/')[-1])
            for i in range(0, NUM_KPTS):
                key_points_coordinate_list.append(
                    {COCO_KEYPOINT_INDEXES[i]: {'x': pose_preds[0][i][0], 'y': pose_preds[0][i][1]}})
                key_points_prob_list.append({COCO_KEYPOINT_INDEXES[i]: {'score': pose_score[0][i][0]}})
            body_pose_score = calculate_pose_score(pose_score)

            single_body_pose_result = {
                'bbox_coordinates': pose_bbox,
                'bbox_confidence_score': pred_score,
                'key_points_coordinate': key_points_coordinate_list,
                'key_points_prob': key_points_prob_list,
                'body_pose_score': body_pose_score
            }
            pose_result.append(single_body_pose_result)

        end_time = time.time()
        pose_result_of_artifact = {'no_of_body_pose_detected': len(pred_boxes),
                                   'pose_result': pose_result,
                                   'time': end_time - start_time
                                   }
        return pose_result_of_artifact

    def result_on_scan_level(self, scan_parent):
        self.qr_code = []
        self.scan_step = []
        self.artifact_id = []
        self.no_of_body_pose_detected = []
        self.pose_result = []
        self.time = []
        # self.artifact_pose_result = []
        logging.info("Extracting artifacts from scans")

        artifact_paths = glob.glob(os.path.join(scan_parent, "**/**/*.jpg"))

        logging.info("Result Generation Started")
        for jpg_path in artifact_paths:
            split_path = jpg_path.split('/')

            qr_code, scan_step, artifact_id = split_path[3], split_path[4], split_path[5]
            pose_result_of_artifact = self.result_on_artifact_level(jpg_path, scan_step)

            self.qr_code.append(qr_code)
            self.scan_step.append(scan_step)
            self.artifact_id.append(artifact_id)
            self.no_of_body_pose_detected.append(pose_result_of_artifact['no_of_body_pose_detected'])
            self.pose_result.append(pose_result_of_artifact['pose_result'])
            self.time.append(pose_result_of_artifact['time'])
            # self.artifact_pose_result.append(pose_result_of_artifact)

    def store_result_in_dataframe(self):
        self.df = pd.DataFrame({
            'scan_id': self.qr_code,
            'scan_step': self.scan_step,
            'artifact_id': self.artifact_id,
            'no_of_body_pose_detected': self.no_of_body_pose_detected,
            'pose_result': self.pose_result,
            'processing_time': self.time
        })

    def save_to_csv(self, file_path):
        self.df.to_csv(file_path, index=False)


def main():
    ctx = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("%s %s", "cuda is available", torch.cuda.is_available())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    args = 'cgmml/models/HRNET/inference-config-hrnet.yaml'
    update_config(cfg, args)

    pose_prediction = PosePrediction(ctx)
    pose_prediction.load_box_model()
    pose_prediction.load_pose_model()

    result_generation = ResultGeneration(pose_prediction, False)
    result_generation.result_on_scan_level('data/anon_rgb_training/scans')

    logging.info("Result Generation done")

    result_generation.store_result_in_dataframe()
    result_generation.save_to_csv(cfg.MODEL.NAME + '.csv')


if __name__ == '__main__':
    main()
