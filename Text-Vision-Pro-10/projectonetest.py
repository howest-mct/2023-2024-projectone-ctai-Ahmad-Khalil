import cv2
import requests
from ultralytics import YOLO

# Define the classes
LETTERS = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]  # A-Z, a-z
NUMBERS = [str(i) for i in range(10)]  # 0-9

# Initialize YOLO model
model = YOLO(r"C:\Users\ahmad\OneDrive\Desktop\2023-2024-projectone-ctai-BigDracco\Text-Vision-Pro-10\best.pt")

def send_message_to_pi(letter_count, number_count):
    url = ' http://192.168.168.167:5000/send-message'  # Replace with the actual IP address of the Raspberry Pi
    payload = {'letter_count': letter_count, 'number_count': number_count}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Message sent: Letters={letter_count}, Numbers={number_count}")
        else:
            print("Failed to send message")
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")

# Capture video from the camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make predictions
        results = model(frame)

        # Extract the counts of letters and numbers
        letter_count = 0
        number_count = 0
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Get confidences

            for i, box in enumerate(boxes):
                class_id = int(class_ids[i])
                label = model.names[class_id]

                if label in LETTERS:
                    letter_count += 1
                elif label in NUMBERS:
                    number_count += 1

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = map(int, box[:4])
                conf = confidences[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send data to Raspberry Pi
        send_message_to_pi(letter_count, number_count)

        # Display the frame with annotations
        cv2.imshow("Object Detection", frame)

        # Check for key press (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Release the camera
cap.release()
cv2.destroyAllWindows()
