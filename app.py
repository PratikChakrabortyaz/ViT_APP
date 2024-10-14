import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()  


def predict(image):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]


interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(type="pil"), 
                         outputs="label", 
                         title="CIFAR-10 Image Classification with ViT")


interface.launch()
