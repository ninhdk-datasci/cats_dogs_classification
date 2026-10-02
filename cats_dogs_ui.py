import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import torch 
from cats_dogs_classification.models.cnn import CNN
from torchvision import transforms
import cv2 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

# from keras.models import load_model
# model = load_model('model1_catsVSdogs_10epoch.h5') 
best_model_path = "./cnn_model/best_checkpoint.pth"
checkpoint = torch.load(best_model_path, map_location=torch.device('cpu')) 
model = CNN(num_classes=2) 
model.load_state_dict(checkpoint["model_state_dict"]) 
#dictionary to label all traffic signs class.
classes = { 
    0:'its a cat',
    1:'its a dog',
 
}
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('CatsVSDogs Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {file_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # [H, W, 3] → [3, 128, 128] → [1, 3, 128, 128]
    x = transform(image).unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
   command=lambda: classify(file_path),
   padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="CatsVSDogs Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()