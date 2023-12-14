# 결과가 불안정하게 출력되며 각막부골편 질환의 경우, 타 질병보다 높은 신뢰도와 정확한 라벨 출력하며 타 질병의 경우는 데이터셋 내의 트레인 사진을 테스트 진행할 경우에만 일부 검출해냄
from torchvision import transforms
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import sys
from torchvision.models import resnet50
import torch.nn as nn
import webbrowser

class_labels = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8"]
class_links = {
    "class name": "link",
}

class MyApp:
    def __init__(self, master):
        self.master = master
        self.master.title("title_name")
        self.master.geometry("842x650")
        self.setup_gui()

    def setup_gui(self):
        # 레이아웃 설정
        box1 = tk.Frame(self.master, width=830, height=500, bg=self.master.cget("bg"), borderwidth=0, relief="solid")
        box1.grid(row=0, column=0, padx=(10, 10), pady=(10, 10))

        # 이미지 경로 관련 위젯 등의 설정 코드
        # 이미지 경로 관련 위젯 설정
        image_path_label = tk.Label(box1, text="but_name:", padx=5, bg="#f0f0f0")
        self.image_path_var = tk.StringVar()
        image_path_entry = tk.Entry(box1, textvariable=self.image_path_var, width=60, state='readonly', bg='white')
        load_image_button = tk.Button(box1, text="but_name", command=self.load_image)
        detect_button = tk.Button(box1, text="but_name", command=self.detect_objects_and_draw)

        image_path_label.grid(row=0, column=0, pady=(10, 10))
        image_path_entry.grid(row=0, column=1, pady=(10, 10))
        load_image_button.grid(row=0, column=2, pady=(10, 10), padx=(5, 5))
        detect_button.grid(row=0, column=3, pady=(10, 10))

        # 이미지 및 결과 표시 관련 위젯 설정
        self.box2 = tk.Frame(self.master, width=400, height=400, bg=self.master.cget("bg"), borderwidth=1, relief="solid")
        self.box2.grid(row=1, column=0, pady=(0, 10), padx=(10, 0))

        self.image_label = tk.Label(self.box2, text="The image you selected will appear here.", bg=self.master.cget("bg"),
                                    width=57, height=26, compound=tk.TOP, relief="solid", borderwidth=1)
        self.image_label.grid(row=0, column=0, padx=(0, 10), pady=(10, 10))

        self.result_image_label = tk.Label(self.box2, text="", bg=self.master.cget("bg"), width=57, height=26,
                                           compound=tk.TOP, relief="solid", borderwidth=1)
        self.result_image_label.grid(row=0, column=1, pady=(10, 10))

        # 객체 감지 결과 정보 표시 관련 위젯 설정
        self.label_info = tk.Label(self.master, text="Disease: ", bg=self.master.cget("bg"))
        self.confidence_info = tk.Label(self.master, text="Confidence: ", bg=self.master.cget("bg"))
        self.additional_info = tk.Label(self.master, text="Additional Information: ", bg=self.master.cget("bg"))

        self.label_info.grid(row=2, column=0, pady=(10, 0))
        self.confidence_info.grid(row=3, column=0, pady=(5, 0))
        self.additional_info.grid(row=4, column=0, pady=(5, 0))

        # 초기화
        self.initial_image_path = None
        self.current_image = None
        self.predicted_class_link = ""
        self.model = None  # 모델 객체 초기화

        # 학습된 모델의 가중치 및 클래스 정보를 로드
        self.load_model_weights_and_class_info()

    def load_model_weights_and_class_info(self):
        try:
            epoch_file_path = "C:/Users/kangk/project/58_ex01_0717_resnet50_checkpoint.pt"
            # 학습된 모델 파일 경로 수정해야함(성능 향상시킨 모델 파일이 있을 경우)
            checkpoint = torch.load(epoch_file_path, map_location=torch.device('cpu'))
            model_class = checkpoint.get('model_class', resnet50)
            
            self.model = model_class(pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 20)
            
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.eval()
            print("Model weights loaded successfully!")

        except Exception as e:
            print("Error loading model weights and class info:", str(e))
            messagebox.showerror("error", "An error occurred while loading model weights and class information.")

    # 이미지 불러오기
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        self.image_path_var.set(file_path)
        self.image_label.config(text="")

        if file_path:
            print("Image Loaded:", file_path)  # 디버깅 포인트
            image = Image.open(file_path)
            image = image.resize((400, 400), Image.LANCZOS)
            self.current_image = image
            photo = ImageTk.PhotoImage(self.current_image)
            self.image_label.configure(image=photo, width=400, height=400)
            self.image_label.image = photo
            self.initial_image_path = file_path
            self.result_image_label.config(text="")        

    def perform_object_detection(self, image_path):
        try:
            print("Loading Eye Disease Model...")
            image_tensor = self.preprocess_image(image_path)

            with torch.no_grad():
                predictions = self.model(image_tensor)

            predicted_class_index = predictions.argmax().item()
            predicted_class_label = class_labels[predicted_class_index]
            print("Predicted Class Index:", predicted_class_index)
            print("Predicted Class Label:", predicted_class_label)
            self.predicted_class_link = class_links.get(predicted_class_label, "https://example.com/default_info")
            if self.predicted_class_link == "https://example.com/default_info":
                print(f"Link not found for class: {predicted_class_label}")


            confidence = str(predictions.softmax(dim=1)[0, predicted_class_index].item())

            result_image = Image.fromarray((image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            result_image = result_image.resize((400, 400), Image.LANCZOS)

            print("Predicted Eye Disease Label:", predicted_class_label)
            print("Confidence:", confidence)

            # 디버깅: 클래스 정보 확인 코드 추가
            print("Class Mapping:", class_labels)
            print("Class Label:", predicted_class_label)
            print("Class Processing:", class_links.get(predicted_class_label, "https://example.com/default_info"))

            return result_image, predicted_class_label, confidence

        except Exception as e:
            print("Error during object detection:", str(e))
            traceback.print_exc()  # 예외를 표시하는 traceback 출력 추가
            messagebox.showerror("error", "text")

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)
        return input_tensor

    def show_additional_info(self):
        webbrowser.open(self.predicted_class_link)

    def detect_objects_and_draw(self):
        if self.initial_image_path:
            print("Performing Object Detection on:", self.initial_image_path)
            
            # 디버깅: 예상치 못한 오류를 찾기 위해 try-except 블록으로 감싸기
            try:
                result_image, label, confidence = self.perform_object_detection(self.initial_image_path)

                self.label_info.config(text="Disease: " + label)
                self.confidence_info.config(text="Confidence: " + confidence)
                self.additional_info.config(text="text", cursor="hand2", fg="blue", underline=True)
                self.additional_info.bind("<Button-1>", lambda e: self.show_additional_info())

                result_photo = ImageTk.PhotoImage(result_image)
                self.result_image_label.configure(image=result_photo, width=400, height=400)
                self.result_image_label.image = result_photo
            
            # 디버깅: 예외 상황을 출력하여 어떤 오류가 발생했는지 확인
            except Exception as e:
                print("Error during object detection and drawing:", str(e))
                messagebox.showerror("error", "An error occurred while performing the diagnostic operation and displaying the results.")

        else:
            messagebox.showwarning("warning", "Load the image first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()









