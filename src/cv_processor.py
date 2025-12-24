import cv2
import numpy as np

class CVProcessor:
    def get_precise_center(self, image_crop):
        """
        Input: Ảnh cắt nhỏ (chỉ chứa 1 linh kiện)
        Output: Tọa độ tâm (cx, cy) tương đối trong ảnh cắt
        """
        # 1. Chuyển sang ảnh xám
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Nhị phân hóa (Thresholding) - Tự động tìm ngưỡng (Otsu)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Tìm Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 5. Lấy contour lớn nhất (giả sử là linh kiện)
        c = max(contours, key=cv2.contourArea)
        
        # 6. Tính Momment để lấy tâm
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    