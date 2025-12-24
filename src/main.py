# File: src/main.py

import cv2
import os
from ai_detector import AIDetector
from cv_processor import CVProcessor

def main():
    # --- CẬP NHẬT ĐƯỜNG DẪN Ở ĐÂY ---
    # Lưu ý: folder 'runs' nằm ngang hàng với 'src' nên dùng ../
    # Bạn nhớ kiểm tra kỹ tên folder là pcb_train_v2 hay v22 nhé (trong log của bạn là v22)
    model_path = '../runs/detect/pcb_train_v22/weights/best.pt' 
    
    print(f"Đang tải model từ: {model_path} ...")
    
    # Kiểm tra file có tồn tại không để tránh lỗi
    if not os.path.exists(model_path):
        print("LỖI: Không tìm thấy file model! Hãy kiểm tra lại đường dẫn.")
        return

    detector = AIDetector(model_path=model_path) 
    processor = CVProcessor()
    
    # --- CHỌN ẢNH ĐỂ TEST ---
    # Bạn hãy copy 1 ảnh mới (chưa từng train) vào folder data/test_images/
    # Ví dụ: đặt tên là test_1.jpg
    img_path = '../data/test_1.jpg' 
    
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh tại: {img_path}")
        # Tạo ảnh tạm để test nếu chưa có
        return

    frame = cv2.imread(img_path)
    
    # --- BẮT ĐẦU QUY TRÌNH HYBRID ---
    # 1. AI Detect (Tìm hộp)
    boxes = detector.detect(frame)
    print(f"AI tìm thấy {len(boxes)} linh kiện.")

    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        
        # 2. Cắt ảnh con
        roi = frame[y1:y2, x1:x2]
        
        # 3. OpenCV (Tìm tâm chính xác trong hộp)
        center_relative = processor.get_precise_center(roi)
        
        # Vẽ hộp (Màu xanh lá)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if center_relative:
            cx_rel, cy_rel = center_relative
            center_absolute = (x1 + cx_rel, y1 + cy_rel)
            
            # Vẽ tâm (Màu đỏ chấm tròn)
            cv2.circle(frame, center_absolute, 5, (0, 0, 255), -1)
            # Ghi toạ độ
            cv2.putText(frame, f"({center_absolute[0]},{center_absolute[1]})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Lưu kết quả
    output_path = '../runs/ket_qua_cuoi_cung.jpg'
    cv2.imwrite(output_path, frame)
    print(f"XONG! Mở file này để xem kết quả: {output_path}")

if __name__ == "__main__":
    main()
# import cv2
# import os
# from ai_detector import AIDetector
# from cv_processor import CVProcessor

# def main():
#     # 1. Khởi tạo
#     # Nếu bạn chưa train, nó sẽ tự tải yolov8n.pt chuẩn (detect người/xe...)
#     # Sau này bạn thay bằng đường dẫn model bạn đã train (vd: 'runs/detect/train/weights/best.pt')
#     detector = AIDetector(model_path='../models/yolov8n.pt') 
#     processor = CVProcessor()
    
#     # 2. Load ảnh (Đường dẫn phải đúng trong Docker)
#     img_path = '../data/images/test_pcb.jpg' # Bạn nhớ bỏ 1 ảnh vào đây
#     if not os.path.exists(img_path):
#         print("Không tìm thấy ảnh!")
#         return

#     frame = cv2.imread(img_path)
#     if frame is None:
#         print("Lỗi đọc ảnh!")
#         return

#     # 3. AI Detect
#     boxes = detector.detect(frame)
#     print(f"AI tìm thấy {len(boxes)} linh kiện.")

#     # 4. Duyệt qua từng linh kiện để tìm tâm chính xác
#     for box in boxes:
#         x1, y1, x2, y2, conf, cls = map(int, box[:6])
        
#         # Cắt ảnh con (ROI - Region of Interest)
#         # Lưu ý: y trước x
#         roi = frame[y1:y2, x1:x2]
        
#         # Xử lý ảnh tìm tâm trong vùng ROI
#         center_relative = processor.get_precise_center(roi)
        
#         if center_relative:
#             cx_rel, cy_rel = center_relative
#             # Tính tâm tuyệt đối trên ảnh gốc
#             center_absolute = (x1 + cx_rel, y1 + cy_rel)
            
#             # Vẽ tâm (Màu đỏ)
#             cv2.circle(frame, center_absolute, 5, (0, 0, 255), -1)
#             # Vẽ tọa độ
#             cv2.putText(frame, f"{center_absolute}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Vẽ khung bao của AI (Màu xanh lá)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # 5. Lưu kết quả
#     output_path = '../runs/result.jpg'
#     cv2.imwrite(output_path, frame)
#     print(f"Đã lưu kết quả tại: {output_path}")

# if __name__ == "__main__":
#     main()