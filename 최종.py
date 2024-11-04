import torch
import pandas as pd

# IoU 기반 Non-Maximum Suppression (NMS) 함수 정의
def non_max_suppression(predictions, iou_threshold=0.5):
    keep_boxes = []
    while predictions.size(0) > 0:
        # 신뢰도가 가장 높은 객체 선택
        highest_confidence_idx = torch.argmax(predictions[:, 4])
        highest_confidence_box = predictions[highest_confidence_idx]
        keep_boxes.append(highest_confidence_box)

        if predictions.size(0) == 1:
            break

        # IoU를 계산하여 신뢰도가 낮은 박스를 제거
        others = predictions[:highest_confidence_idx].tolist() + predictions[highest_confidence_idx + 1:].tolist()
        others = torch.tensor(others)

        ious = bbox_iou(highest_confidence_box[:4], others[:, :4])

        # IoU가 일정 임계값 이하인 박스들만 남김
        predictions = others[ious <= iou_threshold]

    return torch.stack(keep_boxes)

# IoU 계산 함수
def bbox_iou(box1, box2):
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

# YOLOv5 모델 로드 (CPU 환경에서 best.pt 파일 경로)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/근태모델/melody/best.pt', force_reload=True)

# 테스트할 이미지 경로
img_path = 'C:/Users/yangw/OneDrive/바탕 화면/dfdf.jpg'

# 이미지를 모델에 입력하여 예측
results = model(img_path)

# 예측 결과에서 각 객체에 대한 정보
predictions = results.xyxy[0]  # xyxy 형식으로 bbox 정보 가져오기 (x1, y1, x2, y2, confidence, class)

# NMS 적용하여 중복 객체 제거
nms_predictions = non_max_suppression(predictions)

# 클래스 이름 가져오기
class_names = results.names

# 감지된 객체 정보 출력
if len(nms_predictions) > 0:
    for pred in nms_predictions:
        class_id = int(pred[5])
        confidence = pred[4].item()
        class_name = class_names[class_id]
        print(f"Class: {class_name}")
else:
    print("No objects detected.")
