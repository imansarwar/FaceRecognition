import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
from PIL import Image, ImageTk

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Initialize Inception Resnet V1 for face embeddings
model = InceptionResnetV1(pretrained='vggface2').eval()

# Transformation to align the images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")

        self.user_id = StringVar()
        self.embeddings = []
        self.names = []

        # User ID
        Label(root, text="User ID:").grid(row=0, column=0, padx=10, pady=10)
        Entry(root, textvariable=self.user_id).grid(row=0, column=1, padx=10, pady=10)

        # Buttons
        Button(root, text="Take Image", command=self.take_image).grid(row=1, column=0, padx=10, pady=10)
        Button(root, text="Train Images", command=self.train_images).grid(row=1, column=1, padx=10, pady=10)
        Button(root, text="Track Image", command=self.track_image).grid(row=1, column=2, padx=10, pady=10)

        # Video capture
        self.video_capture = cv2.VideoCapture(0)
        self.update_image()

    def update_image(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)
            if boxes is not None:
                for box in boxes:
                    top, right, bottom, left = [int(coord) for coord in box]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            if hasattr(self, 'image_label'):
                self.image_label.config(image=imgtk)
            else:
                self.image_label = Label(self.root, image=imgtk)
                self.image_label.image = imgtk
                self.image_label.grid(row=2, column=0, columnspan=3)
        self.root.after(10, self.update_image)

    def take_image(self):
        user_id = self.user_id.get()
        if not user_id:
            messagebox.showerror("Error", "Please enter a User ID")
            return

        user_folder = os.path.join('dataset', 'training', user_id)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        if count >= 60:
            messagebox.showinfo("Info", "60 images already captured for this user")
            return

        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is not None:
            for box in boxes:
                top, right, bottom, left = [int(coord) for coord in box]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        image_path = os.path.join(user_folder, f'{user_id}_{count+1}.jpg')
        cv2.imwrite(image_path, frame)
        messagebox.showinfo("Info", f"Image saved to {image_path}")

        if count + 1 >= 60:
            messagebox.showinfo("Info", "60 images successfully captured for this user")

    def train_images(self):
        self.embeddings = []
        self.names = []

        for user_id in os.listdir(os.path.join('dataset', 'training')):
            user_folder = os.path.join('dataset', 'training', user_id)
            if os.path.isdir(user_folder):
                for img_name in os.listdir(user_folder):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(user_folder, img_name)
                        image = cv2.imread(img_path)
                        boxes, _ = mtcnn.detect(image)
                        if boxes is not None:
                            faces = mtcnn(image)
                            for face in faces:
                                face = transform(face).unsqueeze(0)
                                embedding = model(face).detach().numpy()
                                self.embeddings.append(embedding)
                                self.names.append(user_id)

        self.embeddings = np.array(self.embeddings)
        messagebox.showinfo("Info", "Training complete")

    def track_image(self):
        if not hasattr(self, 'embeddings'):
            messagebox.showerror("Error", "Please train the model first")
            return

        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        rgb_frame = frame[:, :, ::-1]
        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is not None:
            faces = mtcnn(rgb_frame)
            for face, box in zip(faces, boxes):
                face = transform(face).unsqueeze(0)
                embedding = model(face).detach().numpy()

                distances = np.linalg.norm(self.embeddings - embedding, axis=1)
                min_distance_index = np.argmin(distances)
                name = "Unknown"
                if distances[min_distance_index] < 1.0:
                    name = self.names[min_distance_index]

                top, right, bottom, left = [int(coord) for coord in box]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} ({self.user_id.get()})', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
