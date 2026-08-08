import cv2

# Initialize Pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print('Frame failed')
        break

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    confidence_threshold = 0.7

    for box, weight in zip(boxes, weights):
        if weight > confidence_threshold:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Original Live Feed', frame)
    if cv2.waitKey(1) % 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()