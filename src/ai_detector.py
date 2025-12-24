from ultralytics import YOLO

class AIDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load model, chuyển sang GPU nếu có
        self.model = YOLO(model_path)
    
    def detect(self, image):
        """
        Input: Ảnh gốc
        Output: List các kết quả (xyxy coordinates, class_id, conf)
        """
        # predict return list, lấy phần tử đầu tiên
        results = self.model(image, verbose=False)[0] 
        boxes = results.boxes.data.cpu().numpy() # Chuyển về numpy array
        return boxes # Trả về mảng [x1, y1, x2, y2, conf, cls]