import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2) 
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2) 

class ViT(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.transformer_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.transformer_layers:
            x = layer(x) + x 
        x = x.mean(dim=1)  
        return self.classifier(x)


model = ViT()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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


