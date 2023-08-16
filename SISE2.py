import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
from utils import *
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import gc

class SISE():
    def __init__(self, model, model_name, img_path, class_idx, device='cuda', grouping_thr=0.5, detail=0):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.img = Image.open(img_path).resize((224, 224))
        img_arr = np.asarray(self.img)[:, :, :3] / 255.
        self.input_img = torch.tensor(img_arr).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
        self.class_idx = class_idx
        self.grouping_thr = grouping_thr
        self.detail = detail
        self.model.to(device).eval()

    def otsu_threshold(self, tensor):
        hist = torch.histc(tensor, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        bin_centers = torch.linspace(0, 1, 256)
        weight1 = hist.cumsum(0)
        weight2 = 1 - weight1
        mean1 = (hist * bin_centers).cumsum(0) / weight1
        mean2 = (hist * bin_centers).sum() - mean1
        variance = weight1 * weight2 * (mean1 - mean2) ** 2
        idx = variance.argmax()
        return bin_centers[idx].item()

    def otsu_binary(self, tensor):
        threshold = self.otsu_threshold(tensor)
        return tensor > threshold

    def IoU(self, bbox1, bbox2):
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)
        bbox1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        bbox2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
        union_area = bbox1_area + bbox2_area - inter_area
        iou = inter_area / union_area
        return iou

    def normalization(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def feature_extractor(self):
        layers = {
            'vgg16': ['5', '10', '17', '24', '31', '35'],
            'resnet50': ['bn1', 'layer1.0.downsample.1', 'layer2.0.downsample.1', 'layer3.0.downsample.1', 'layer4.0.downsample.1'],
            'resnet152': ['bn1', 'layer1.0.downsample.1', 'layer2.0.downsample.1', 'layer3.0.downsample.1', 'layer4.0.downsample.1']
        }

        block = layers[self.model_name]

        feature_maps = []
        hooks = []
        for name, module in self.model.named_modules():
            if name in block:
                hook = module.register_forward_hook(lambda module, input, output: feature_maps.append(output))
                hooks.append(hook)
        
        _ = self.model(self.input_img)

        for hook in hooks:
            hook.remove()

        self.feature_maps = feature_maps

    def feature_filtering(self):
        layers = {
            'vgg16': ['5', '10', '17', '24', '31', '35'],
            'resnet50': ['bn1', 'layer1.0.downsample.1', 'layer2.0.downsample.1', 'layer3.0.downsample.1', 'layer4.0.downsample.1'],
            'resnet152': ['bn1', 'layer1.0.downsample.1', 'layer2.0.downsample.1', 'layer3.0.downsample.1', 'layer4.0.downsample.1']
        }

        block = layers[self.model_name]
        grads = []
        hooks = []

        for name, module in self.model.named_modules():
            if name in block:
                hook = module.register_backward_hook(lambda module, grad_input, grad_output: grads.append(grad_output[0]))
                hooks.append(hook)
        
        self.model.zero_grad()
        self.input_img.requires_grad = True
        outputs = self.model(self.input_img)
        one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_().to(self.device)
        one_hot_output[0][self.class_idx] = 1
        outputs.backward(gradient=one_hot_output)

        for hook in hooks:
            hook.remove()

        grads.reverse()
        self.avg_grads = [torch.mean(grad[0], dim=[1,2]) for grad in grads]
        self.filtered_feature_maps = [fmap[:,grad>0] for fmap, grad in zip(self.feature_maps, self.avg_grads)]
        
    def postprocess(self):
        postprocessed_feature_maps = []
        for fmap in self.filtered_feature_maps:
            fmap = fmap.permute(0, 2, 3, 1)  # Changing the order to NHWC
            processed_fmap = []
            for channel in range(fmap.shape[-1]):
                img = fmap[0, :, :, channel]
                img = img.unsqueeze(0).unsqueeze(0)  # Adding N and C dimensions
                img_resized = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                img_resized = img_resized.squeeze()  # Removing N and C dimensions
                img_min = torch.min(img_resized)
                img_max = torch.max(img_resized)
                img_normalized = (img_resized - img_min) / (img_max - img_min)
                processed_fmap.append(img_normalized)

            processed_fmap_stack = torch.stack(processed_fmap, dim=2)
            postprocessed_feature_maps.append(processed_fmap_stack)

        self.postprocessed_feature_maps = postprocessed_feature_maps
        # postprocessed_feature_maps = []
        # for fmap in self.filtered_feature_maps:
        #     fmap = fmap.permute(0,2,3,1).detach().cpu().numpy()
        #     processed_fmap = []
        #     for channel in range(fmap.shape[-1]):
        #         img = cv2.resize(fmap[0,:,:,channel], (224,224), interpolation=cv2.INTER_LINEAR)
        #         img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #         processed_fmap.append(img)
        #     postprocessed_feature_maps.append(np.stack(processed_fmap, axis=2))
        # self.postprocessed_feature_maps = postprocessed_feature_maps
    
    def filtering_zero_feature_maps(self):
        sum_feature_maps = [torch.sum(fmap[0], dim=(1,2)) for fmap in self.filtered_feature_maps]
        self.filtered_feature_maps = [fmap[:,sum_fmap!=0] for fmap, sum_fmap in zip(self.filtered_feature_maps, sum_feature_maps)]
    
    def attribution_masks_compress(self):
        layers = [3, 4]

        layer_bbox = {}

        # for layer in layers:
        #     layer_bbox[layer] = []
        #     for index in range(self.postprocessed_feature_maps[layer].shape[2]):
        #         binary = otsu_binary(self.postprocessed_feature_maps[layer][:,:,index])
        #         labeled, nr_objects = label(binary > 0)
        #         props = regionprops(labeled)

        #         init = props[0].bbox_area
        #         bbox = tuple(props[0].bbox)
        #         for b in props:
        #             if init < b.bbox_area:
        #                 init = b.bbox_area
        #                 bbox = tuple(b.bbox)

        #         layer_bbox[layer].append(bbox)

        for layer in layers:
            layer_bbox[layer] = []
            for index in range(self.postprocessed_feature_maps[layer].shape[2]):
                binary = self.otsu_binary(self.postprocessed_feature_maps[layer][:,:,index].cpu()).numpy()
                labeled, nr_objects = label(binary > 0)
                props = regionprops(labeled)
                
                init = props[0].bbox_area
                bbox = tuple(props[0].bbox)
                for b in props:
                    if init < b.bbox_area:
                        init = b.bbox_area
                        bbox = tuple(b.bbox)

                layer_bbox[layer].append(bbox)

        group_bbox = {}
        for k in layer_bbox.keys():
            temp = layer_bbox[k].copy()
            group_bbox[k] = []
            for i in range(len(temp)):
                if temp[i] == 0:
                    continue
                temp_group = [i]
                for j in range(i+1, len(temp)):
                    if temp[j] == 0:
                        continue
                    if self.IoU(temp[i], temp[j]) >= self.grouping_thr:
                        temp_group.append(j)
                        temp[j] = 0
                temp[i] = 0
                group_bbox[k].append(temp_group)

        self.group_bbox = group_bbox

        compressed_feature_maps = {}
        for layer in layers:
            for b in group_bbox[layer]:
                compressed_feature_map = torch.zeros_like(self.postprocessed_feature_maps[layer][:,:,0])
                for i, feature_map_index in enumerate(b):
                    if i == 0:
                        compressed_feature_map = self.postprocessed_feature_maps[layer][:,:, feature_map_index]
                    else:
                        compressed_feature_map += self.postprocessed_feature_maps[layer][:,:, feature_map_index]
                if layer in compressed_feature_maps:
                    compressed_feature_map = self.normalization(compressed_feature_map)
                    compressed_feature_maps[layer] = torch.cat((compressed_feature_maps[layer], compressed_feature_map.unsqueeze(2)), dim=2)
                else:
                    compressed_feature_map = self.normalization(compressed_feature_map)
                    compressed_feature_maps[layer] = compressed_feature_map.unsqueeze(2)

            self.postprocessed_feature_maps[layer] = compressed_feature_maps[layer]

        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads, self.postprocessed_feature_maps):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]
        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')
        self.total_reduction_rate = (sum2-sum1)/sum1*100

    def new_attribution_masks_compress1(self, mode):
        layers = [3, 4]

        layer_bbox = {}

        # conv3, conv4 layer's feature maps bbox coordinate calculation
        for layer in layers:
            layer_bbox[layer] = []
            for index in range(self.postprocessed_feature_maps[layer].shape[2]):
                binary = otsu_binary(self.postprocessed_feature_maps[layer][:,:,index])
                layer_bbox[layer].append(binary)

        # Grouping feature maps with IoU more than 0.5
        group_bbox = {}
        for k in layer_bbox.keys():
            temp = layer_bbox[k].copy()
            group_bbox[k] = []
            for i in range(len(temp)):
                if not isinstance(temp[i], torch.Tensor):
                    continue
                temp_group = [i]
                for j in range(i+1, len(temp)):
                    if not isinstance(temp[j], torch.Tensor):
                        continue
                    if bitwiseSimilarity(temp[i], temp[j], mode) >= self.grouping_thr:
                        temp_group.append(j)
                        temp[j] = 0
                temp[i] = 0
                group_bbox[k].append(temp_group)

        self.group_bbox = group_bbox

        compressed_feature_maps = {}

        for layer in layers:
            for b in group_bbox[layer]:
                compressed_feature_map = torch.zeros_like(self.postprocessed_feature_maps[layer][:,:,0].shape)
                for i, feature_map_index in enumerate(b):
                    if i == 0:
                        compressed_feature_map = self.postprocessed_feature_maps[layer][:,:, feature_map_index]
                    else:
                        compressed_feature_map += self.postprocessed_feature_maps[layer][:,:, feature_map_index]

                if layer in compressed_feature_maps:
                    compressed_feature_maps[layer] = torch.cat((compressed_feature_maps[layer], torch.unsqueeze(normalization(compressed_feature_map), dim=2)), dim=2)
                else:
                    compressed_feature_maps[layer] = torch.unsqueeze(normalization(compressed_feature_map), dim=2)

            self.postprocessed_feature_maps[layer] = compressed_feature_maps[layer]

        # Comparing the number of filtered feature maps
        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads.values(), self.postprocessed_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]

        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')
        self.total_reduction_rate = (sum2-sum1)/sum1*100

    # similar translations for the rest of the methods
    def generate_layer_visualization_map(self, batch_size=4):
        layer_visualization_maps = []
        
        with torch.no_grad():
            for fmap in self.postprocessed_feature_maps:
                masks = fmap.permute(2, 0, 1).unsqueeze(1)
                masked = self.input_img * masks
                preds = self.model(masked)
                layer_visualization_maps.append((preds.T @ masks.view(masks.shape[0], -1)).view(-1, 224, 224))

            self.layer_visualization_maps = layer_visualization_maps

    def layers_fusion(self):
        result = self.normalization(self.layer_visualization_maps[0][self.class_idx]).clone()

        for i, _ in enumerate(self.layer_visualization_maps):
            if i == 0:
                continue
            result += self.layer_visualization_maps[i][self.class_idx]
            thr = filters.threshold_otsu(self.normalization(self.layer_visualization_maps[i][self.class_idx]).cpu().numpy())
            binary = self.normalization(self.layer_visualization_maps[i][self.class_idx]) > thr
            binary = binary * 255
            result = result * binary

        self.result = result