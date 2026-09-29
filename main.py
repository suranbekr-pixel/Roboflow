import warnings
warnings.filterwarnings("ignore")

import cv2
from dotenv import load_dotenv
import os
from inference_sdk import InferenceHTTPClient

load_dotenv()

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv('API_KEY')
)

confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.40'))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Камера жок')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Кадр жок')
        break

    sheep_count = 0
    cow_count = 0

    results = client.infer(frame, model_id=os.getenv('MODEL_ID'))

    for i in results.get('predictions', []):
        confidence = i.get('confidence', 0)

        if confidence < confidence_threshold:
            continue

        x = i['x']
        y = i['y']
        width = i['width']
        height = i['height']
        class_name = i['class']

        x1 = int(x - (width / 2))
        y1 = int(y - (height / 2))
        x2 = int(x + (width / 2))
        y2 = int(y + (height / 2))

        if class_name.lower() == 'sheep':
            sheep_count += 1
            color = (0, 0, 255)
        elif class_name.lower() == 'cow':
            cow_count += 1
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)


        conf_percent = int(round(confidence * 100))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name}, {conf_percent}%', (x1, y1 - 10),
                    cv2.FONT_ITALIC, 0.8, (255, 0, 255), 2)

    cv2.putText(frame, f'SHEEP COUNT: {sheep_count}', (20, 35),
                cv2.FONT_ITALIC, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'COW COUNT: {cow_count}', (20, 70),
                cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Animals detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
