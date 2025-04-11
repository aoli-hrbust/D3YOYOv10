import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def show_cam_on_image1(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:

    mask = np.clip(mask * 2.0, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):

    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class yolov10_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
        model.load_state_dict(csd, strict=False)
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)


        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        post_result, pre_post_boxes, post_boxes = self.post_process(result['one2many'][0])

        max_conf_idx = torch.argmax(post_result.max(1)[0])
        max_conf = float(post_result[max_conf_idx].max())

        if max_conf < self.conf_threshold:
            print("No object meets confidence threshold.")
            return

        self.model.zero_grad()

        if self.backward_type == 'class' or self.backward_type == 'all':
            score = post_result[max_conf_idx].max()
            score.backward(retain_graph=True)

        if self.backward_type == 'box' or self.backward_type == 'all':
            for j in range(4):
                score = pre_post_boxes[max_conf_idx, j]
                score.backward(retain_graph=True)

        if self.backward_type == 'class':
            gradients = grads.gradients[0]
        elif self.backward_type == 'box':
            gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
        else:
            gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                        grads.gradients[4]

        b, k, u, v = gradients.size()
        weights = self.method.get_cam_weights(self.method, None, None, None, activations, gradients.detach().numpy())
        weights = weights.reshape((b, k, 1, 1))
        saliency_map = np.sum(weights * activations, axis=1)

        saliency_map = np.squeeze(np.maximum(saliency_map, 0))
        saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
        saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)), interpolation=cv2.INTER_CUBIC)

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        if (saliency_map_max - saliency_map_min) == 0:
            return

        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        cam_image = show_cam_on_image1(img.copy(), saliency_map, use_rgb=True, image_weight=0.6)
        cam_image = Image.fromarray(cam_image)
        cam_image.save(f'{save_path}/max_conf.png')  # **只保存一张最高置信度的热力图**


def get_params():
    params = {
        'weight': 'weights/best.pt',  # 训练出来的权重文件
        'cfg': 'yolov10s.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM , 使用的热力图库文件不同的效果不一样可以多尝试
        'layer': 'model.model[16]',  # 想要检测的对应层
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.8,  # 0.6  # 置信度阈值，有的时候你的进度条到一半就停止了就是因为没有高于此值的了
        'ratio': 0.02  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    model = yolov10_heatmap(**get_params())
    model(r'images/tomato.jpg', 'runs/detect/')  # 第一个是检测的文件, 第二个是保存的路径