import socket
import cv2
from ultralytics import YOLO

# Define the classes
CLASSES = ["class1", "class2", ..., "class36"]

def send_data_to_pi(letter_count, number_count, host, port):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        data = f"letters:{letter_count},numbers:{number_count}"
        client_socket.sendall(data.encode())
        client_socket.close()
    except Exception as e:
        print("Error sending data to Raspberry Pi:", e)

def detect_objects_and_send_data():
    # Initialize YOLO model
    model = YOLO(r"C:\Users\ahmad\OneDrive\Desktop\2023-2024-projectone-ctai-BigDracco\Text-Vision-Pro-10\best.pt")

    # Open camera feed
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Extract the counts of letters and numbers
        letter_count = 0
        number_count = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Check if the detected object is a letter or a number
                if label.isalpha():
                    letter_count += 1
                elif label.isdigit():
                    number_count += 1

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {box.conf[0]:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Send data to Raspberry Pi
        send_data_to_pi(letter_count, number_count, '192.168.168.167', 5000)  # Replace with Raspberry Pi IP

        # Check for key press (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects_and_send_data()
