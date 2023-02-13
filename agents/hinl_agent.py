import random
import sys

import cv2
import time
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

from robothor_challenge.agent import Agent
from robothor_challenge.challenge import ALLOWED_ACTIONS
from submodules.detr.apis.inference import (inference_detector, init_detector,
                                            make_detr_transforms)
from submodules.detr.util.parse import get_args_parser
from agents.hinl.vtn_visual_embedding import VTRGBVisualEmbedding
from agents.hinl.hinl_navigation_policy import ForgetAttnLSTMNavigationPolicyArchived

TARGETS = [
    'AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan', 'HousePlant',
    'Laptop', 'Mug', 'RemoteControl', 'SprayBottle', 'Television', 'Vase',
]

ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Stop']
# ACTIONS = ['RotateLeft', 'RotateRight', 'MoveAhead', 'LookUp', 'LookDown', 'Stop']

class HiNLAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

        argv = sys.argv
        sys.argv = []
        args = get_args_parser().parse_args()
        args.dataset_file = "RoboTHOR_Detection_Data"
        sys.argv = argv
        self.transform = make_detr_transforms('val')
        self.local_detector, self.detector_postprocessor = init_detector(args)
        detector_state_dict = torch.load('./model_zoo/robothor.13cls.checkpoint0059.pth', map_location='cpu')
        self.local_detector.load_state_dict(detector_state_dict['model'])
        for param in self.local_detector.parameters():
            param.requires_grad = False

        # Extract features from the last two layers
        self.global_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            transforms.ConvertImageDtype(torch.float),
        ])
        resnet18 = models.resnet18(pretrained=True)
        self.resent_extractor = torch.nn.Sequential(*list(resnet18.children())[:-2]).cuda()
        for param in self.resent_extractor.parameters():
            param.requires_grad = False

        self.visual_embedding = VTRGBVisualEmbedding().cuda()
        self.navigation_policy = ForgetAttnLSTMNavigationPolicyArchived().cuda()

        self.last_action_probs = torch.zeros((1, 6)).cuda()
        self.hidden = None
        self.n_steps = 0

        navigation_model = torch.load('./model_zoo/HSN_HSE_HSR_RoboTHOR_mk3_0003900000_2022-11-04_11-10-54.dat', map_location='cpu')
        visual_embedding_state_dict = {k.replace('visual_embedding.', ''):navigation_model['model'][k] for k in navigation_model['model'] if 'visual_embedding' in k}
        self.visual_embedding.load_state_dict(visual_embedding_state_dict)
        navigation_policy_state_dict = {k.replace('navigation_policy.', ''):navigation_model['model'][k] for k in navigation_model['model'] if 'navigation_policy' in k}
        self.navigation_policy.load_state_dict(navigation_policy_state_dict)

    def process_local_feature(self, result, output, target):
        current_detection_feature = torch.cat(
            (
                output['encoder_features'], 
                output['pred_logits'].max(-1)[0].unsqueeze(dim=-1), 
                output['pred_logits'].max(-1)[1].unsqueeze(dim=-1), 
                result[0]['boxes'].unsqueeze(dim=0)
            ),
            dim=-1
        )

        zero_detect_feats = torch.zeros_like(current_detection_feature)
        ind = 0
        for cate_id in range(14):
            cate_index = current_detection_feature[..., 257] == cate_id
            if cate_index.sum() > 0:
                # index = current_detection_feature[cate_index, 256].argmax(0)
                try:
                    zero_detect_feats[:, ind:ind+cate_index.sum(), :] = current_detection_feature[cate_index, :]
                except:
                    print('something wrong1')
                ind += cate_index.sum()
        current_detection_feature = zero_detect_feats

        detection_inputs = {
            'features': current_detection_feature[..., :256],
            'scores': current_detection_feature[..., 256, None],
            'labels': current_detection_feature[..., 257, None],
            'bboxes': current_detection_feature[..., -4:],
            'target': TARGETS.index(target),
        }

        # generate target indicator array based on detection results labels
        target_embedding_array = torch.zeros_like(detection_inputs['labels'])
        target_embedding_array[detection_inputs['labels'] == (TARGETS.index(target) + 1)] = 1
        detection_inputs['indicator'] = target_embedding_array

        return detection_inputs


    def draw_detr_output(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Draw DETR-style outputs on an image.
        
        Arguments:
            image {np.ndarray} -- The input image.
            boxes {np.ndarray} -- The bounding boxes.
            labels {np.ndarray} -- The labels for each object.
            scores {np.ndarray} -- The scores for each object.
        
        Returns:
            np.ndarray -- The image with the DETR-style outputs drawn on it.
        """
        
        # Convert the image from RGB to BGR format
        image = image[..., ::-1]

        # Loop through all the bounding boxes
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            
            # Get the (xmin, ymin, xmax, ymax) coordinates for the box
            xmin, ymin, xmax, ymax = box.astype(int)
            
            # Draw the bounding box on the image
            image = cv2.resize(image,(300,300))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # Create the label string with the label and score
            label = f"{labels[i]}: {scores[i]:0.2f}"
            
            # Determine the position of the text based on the location of the bounding box
            y = ymin - 10 if ymin > 20 else ymin + 10
            
            # Draw the label and score on the image
            cv2.putText(image, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Return the image with the DETR-style outputs drawn on it
        return image

    def reset(self):
        
        self.last_action_probs = torch.zeros((1, 6)).cuda()
        self.hidden = None
        self.n_steps = 0

    def act(self, observations):
        rgb = observations["rgb"]           # np.uint8 : 480 x 640 x 3
        depth = observations["depth"]       # np.float32 : 480 x 640 (default: None)
        goal = observations["object_goal"]  # str : e.g. "AlarmClock"

        output = inference_detector(self.local_detector, self.transform(rgb.copy()).cuda().unsqueeze(dim=0), None)
        result = self.detector_postprocessor['bbox'](output, torch.tensor([[300, 300]]).cuda())
        # detected_result = self.draw_detr_output(
        #     rgb.copy(), result[0]['boxes'].cpu().numpy(), result[0]['labels'].cpu().numpy(), result[0]['scores'].cpu().numpy())
        # cv2.imwrite(f'./images/detection_results/{time.time():.2f}.jpg', detected_result)
        detection_inputs = self.process_local_feature(result, output, goal)

        global_features = self.resent_extractor(self.global_transform(rgb.copy()).unsqueeze(0).cuda())

        visual_inputs = {
            'detection_inputs': detection_inputs,
            'state': global_features,
        }
        visual_outputs = self.visual_embedding(visual_inputs)

        navigation_policy_inputs = {
            'visual_representation': visual_outputs['visual_representation'],
            'last_action': self.last_action_probs,
            'policy_hidden_states': self.hidden,
            'n_steps': self.n_steps,
        }
        navigation_policy_outputs = self.navigation_policy(navigation_policy_inputs)
        self.hidden = navigation_policy_outputs['policy_hidden_states']
        self.last_action_probs = F.softmax(navigation_policy_outputs['actor_output'], dim=-1)

        action = self.last_action_probs.multinomial(1).data
        action = ACTIONS[action]
        # action = random.choice(ALLOWED_ACTIONS)
        return action


def build():
    agent_class = HiNLAgent
    agent_kwargs = {
        # 'hinl_config': './configs/hinl.yaml',
    }
    # resembles SimpleRandomAgent(**{})
    render_depth = False
    return agent_class, agent_kwargs, render_depth
