import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib import patches

class CAM:
    def __init__(self, cam_model) -> None:
        self.cam_model = cam_model

    def simple_cam(self, img_path):
        img = Image.open(img_path).resize((224, 224))
        img_arr = np.asarray(img)[:, :, :3] / 255.
        img_array = np.expand_dims(img_arr, 0)
        
        conv_outputs, prediction = self.cam_model.predict(img_array) # CAM 생성에 필요한 값들
        # prediction = np.argmax(prediction, axis=1)
        class_weights = self.cam_model.layers[-1].get_weights()[0]

        cam = np.dot(conv_outputs[0, :, :, :], class_weights[:,np.argmax(prediction, axis=1)])
        
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = cv2.resize(cam_img, (224, 224))
        cam_img = np.uint8(255 * cam_img)
        
        return img_arr, cam_img, prediction

    def avg_cam(self, img_path):
        img = Image.open(img_path).resize((224, 224))
        img_arr = np.asarray(img)[:, :, :3] / 255.
        img_array = np.expand_dims(img_arr, 0)
        
        conv_outputs, prediction = self.cam_model.predict(img_array) # CAM 생성에 필요한 값들
        class_num = prediction.shape[1]
        prediction = np.argmax(prediction, axis=1)
        class_weights = self.cam_model.layers[-1].get_weights()[0]
        
        ## calculate cam
        cam = np.zeros(conv_outputs.shape[1:3], dtype=float)
        for class_idx in range(1, class_num):
            cam += np.dot(conv_outputs, class_weights[:,class_idx])
        
        cam /= class_num - 1
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = cv2.resize(cam_img, (224, 224))
        cam_img = np.uint8(255*cam_img)
        
        return img_arr, cam_img, prediction

    def back_remove_avg_cam(self, img_path):  
        # Input image resize
        img = Image.open(img_path).resize((224, 224))
        img_arr = np.asarray(img)[:, :, :3] / 255.
        img_array = np.expand_dims(img_arr, 0)

        conv_outputs, prediction = self.cam_model.predict(img_array) # CAM 생성에 필요한 값들
        class_num = prediction.shape[1]
        # prediction = np.argmax(prediction, axis=1)
        class_weights = self.cam_model.layers[-1].get_weights()[0]
        
        ## calculate cam
        cam = np.zeros(conv_outputs.shape[1:3], dtype=float)
        for class_idx in range(1, class_num):
            cam += np.dot(conv_outputs[0, :, :, :], class_weights[:,class_idx])

        # AvgCAM
        cam /= class_num - 1

        # BR-AvgCAM
        back_cam = np.dot(conv_outputs[0, :, :, :], class_weights[:,0])
        cam = cam - back_cam # Back CAM remove
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = cv2.resize(cam_img, (224, 224))
        cam_img = np.uint8(255*cam_img)

        return img_arr, cam_img, prediction

    def back_remove_cam(self, img_path):  
        img = Image.open(img_path).resize((224, 224))
        img_arr = np.asarray(img)[:, :, :3] / 255.
        img_array = np.expand_dims(img_arr, 0)

        conv_outputs, prediction = self.cam_model.predict(img_array) # CAM 생성에 필요한 값들
        # prediction = np.argmax(prediction, axis=1)
        class_weights = self.cam_model.layers[-1].get_weights()[0]

        ## calculate cam
        cam = np.dot(conv_outputs[0, :, :, :], class_weights[:,np.argmax(prediction)])
        back_cam = np.dot(conv_outputs[0, :, :, :], class_weights[:,0])

        cam = cam - back_cam
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = cv2.resize(cam_img, (224, 224))
        cam_img = np.uint8(255*cam_img)

        return img_arr, cam_img, prediction

def get_label(xml):
    p_size = xml.find('size')
    p_box = xml.find('object').find('bndbox')
    size = {'width':int(p_size.find('width').text),'height': int(p_size.find('height').text)}
    box = {'xmin':int(p_box.find('xmin').text), 'ymin' : int(p_box.find('ymin').text),'xmax': int(p_box.find('xmax').text),'ymax': int(p_box.find('ymax').text)}
    xmin, ymin, xmax, ymax = box['xmin'] / size['width'] * 224, box['ymin'] / size['height'] * 224, box['xmax'] / size['width'] * 224,box['ymax'] / size['height'] * 224
    w, h = xmax - xmin, ymax - ymin
    return {'xmin':xmin, 'ymin':ymin, 'xmax':xmax,'ymax':ymax,'w':w, 'h':h}

def generate_bbox(img, cam, threshold):
    labeled, nr_objects = label(cam > threshold)
    props = regionprops(labeled)

    init = props[0].bbox_area
    bbox = list(props[0].bbox)
    for b in props:
        if init < b.bbox_area:
            bbox = list(b.bbox)
    return bbox

#boxA[0] : min x, boxA[1] : min y, boxA[2] : max x, boxA[3] : max y
def IoU(boxA, boxB):
    xA = max(boxA[1], boxB['xmin'])
    yA = max(boxA[0], boxB['ymin'])
    xB = min(boxA[3], boxB['xmax'])
    yB = min(boxA[2], boxB['ymax'])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def avg_blur(bbox, img):
    x = bbox[1]
    y = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    roi = img[y:y+h, x:x+w]
    roi = cv2.blur(roi, (9,9))
    img[y:y+h, x:x+w] = roi

    return img

def gaussian_blur(bbox, img):
    x = bbox[1]
    y = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    roi = img[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (9,9), 0)
    img[y:y+h, x:x+w] = roi

    return img

def median_blur(bbox, img):
    x = bbox[1]
    y = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    roi = img[y:y+h, x:x+w]
    roi = cv2.medianBlur(roi.astype(np.uint8), 9)
    img[y:y+h, x:x+w] = roi

    return img

def bilateral_filter(bbox, img):
    x = bbox[1]
    y = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    roi = img[y:y+h, x:x+w]
    roi = cv2.bilateralFilter(roi.astype(np.float32), 9, 75, 75)
    img[y:y+h, x:x+w] = roi

    return img

def show(img, cam, box, bbox):
    plt.axis('off')
    plt.figure(figsize=(5,5))
    plt.imshow(cam, cmap='jet', alpha=0.9)
    plt.imshow(img, alpha=0.5)
    ax = plt.gca()

    xs = bbox[1]
    ys = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    rect = patches.Rectangle((xs, ys), w, h, linewidth=4, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle((box['xmin'], box['ymin']), box['w'], box['h'], linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    plt.show()