
from HotMap import yolov10_heatmap
from ultralytics import YOLOv10

def get_params():
    params = {
        'weight': 'weights/best.pt',
        'cfg': 'yolov10s.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM',
        'layer': 'model.model[16]',
        'backward_type': 'all',
        'conf_threshold': 0.8,
        'ratio': 0.02
    }
    return params

if __name__ == '__main__':
    model = YOLOv10('weights/best.pt') # select your model.pt path
    model.predict(source='images/tomato.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )

    model_hot = yolov10_heatmap(**get_params())
    model_hot(r'images/tomato.jpg', 'runs/hotmap')