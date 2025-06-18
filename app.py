from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import io
import base64
from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the model architecture (same as in your training script)
class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, X):
        n = X.size(2) * X.size(3) - 1
        mean = X.mean(dim=[2, 3])
        d = (X - mean.unsqueeze(2).unsqueeze(3)).pow(2)
        v = d.sum(dim=[2, 3]) / n
        E_inv = d / (4 * (v + self.lambda_val).unsqueeze(2).unsqueeze(3)) + 0.5
        return X * F.sigmoid(E_inv)

class DWConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConv3x3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, X):
        X = self.depthwise(X)
        X = self.pointwise(X)
        return X

class MCCRM(nn.Module):
    def __init__(self, in_channels):
        super(MCCRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.dwconv1 = DWConv3x3(in_channels, in_channels)
        self.simam = SimAM()
        self.projection = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, X):
        F = X
        F1 = self.conv1(F)
        F2 = self.dwconv1(F1) + F
        G = torch.cat([F, F2], dim=1)
        G = self.simam(G)
        H = self.projection(G)
        return H

class EMPT(nn.Module):
    def __init__(self, dim, num_heads=2):
        super(EMPT, self).__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.ffn_ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_ln = self.ln(x_flat)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        attn_out = attn_out + x_flat
        out = self.ffn_ln(attn_out)
        out = self.ffn(out) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out

class IMTM(nn.Module):
    def __init__(self, dim=128, num_heads=2):
        super(IMTM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.proj1 = nn.Conv2d(32, dim, kernel_size=1)
        self.proj2 = nn.Conv2d(64, dim, kernel_size=1)
        self.proj3 = nn.Conv2d(96, dim, kernel_size=1)
        self.proj4 = nn.Conv2d(128, dim, kernel_size=1)
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.mlp_ln = nn.LayerNorm(dim)
        self.out_proj1 = nn.Conv2d(dim, 32, kernel_size=1)
        self.out_proj2 = nn.Conv2d(dim, 64, kernel_size=1)
        self.out_proj3 = nn.Conv2d(dim, 96, kernel_size=1)
        self.out_proj4 = nn.Conv2d(dim, 128, kernel_size=1)

    def forward(self, t1, t2, t3, t4):
        B, _, H, W = t4.shape
        orig_sizes = [t1.shape[2:], t2.shape[2:], t3.shape[2:], t4.shape[2:]]
        t1 = self.proj1(t1)
        t2 = self.proj2(t2)
        t3 = self.proj3(t3)
        t4 = self.proj4(t4)
        t1 = F.interpolate(t1, size=(H, W), mode='bilinear', align_corners=False)
        t2 = F.interpolate(t2, size=(H, W), mode='bilinear', align_corners=False)
        t3 = F.interpolate(t3, size=(H, W), mode='bilinear', align_corners=False)
        feats = [rearrange(t, 'b c h w -> b (h w) c') for t in [t1, t2, t3, t4]]
        feats = torch.cat(feats, dim=1)
        feats = self.ln(feats)
        attn_out, _ = self.attn(feats, feats, feats)
        attn_out = attn_out + feats
        out = self.mlp_ln(attn_out)
        out = self.mlp(out) + out
        out = out.chunk(4, dim=1)
        out = [rearrange(o, 'b (h w) c -> b c h w', h=H, w=W) for o in out]
        out[0] = F.interpolate(self.out_proj1(out[0]), size=orig_sizes[0], mode='bilinear', align_corners=False)
        out[1] = F.interpolate(self.out_proj2(out[1]), size=orig_sizes[1], mode='bilinear', align_corners=False)
        out[2] = F.interpolate(self.out_proj3(out[2]), size=orig_sizes[2], mode='bilinear', align_corners=False)
        out[3] = F.interpolate(self.out_proj4(out[3]), size=orig_sizes[3], mode='bilinear', align_corners=False)
        return out

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(MCCRM(32))
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(MCCRM(64))
        self.downsample2 = nn.Conv2d(64, 96, kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(MCCRM(96))
        self.downsample3 = nn.Conv2d(96, 128, kernel_size=2, stride=2)
        self.stage4 = nn.Sequential(MCCRM(128))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.stage1(x)
        x2 = self.stage2(self.downsample1(x1))
        x3 = self.stage3(self.downsample2(x2))
        x4 = self.stage4(self.downsample3(x3))
        return x1, x2, x3, x4

class AuxiliaryNetwork(nn.Module):
    def __init__(self):
        super(AuxiliaryNetwork, self).__init__()
        self.empt1 = EMPT(32)
        self.empt2 = EMPT(64)
        self.empt3 = EMPT(96)
        self.empt4 = EMPT(128)

    def forward(self, x1, x2, x3, x4):
        t1 = self.empt1(x1)
        t2 = self.empt2(x2)
        t3 = self.empt3(x3)
        t4 = self.empt4(x4)
        return t1, t2, t3, t4

class IdentityMappingResFormer(nn.Module):
    def __init__(self, num_classes=4):
        super(IdentityMappingResFormer, self).__init__()
        self.backbone = Backbone()
        self.auxiliary = AuxiliaryNetwork()
        self.imtm = IMTM(dim=128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        t1, t2, t3, t4 = self.auxiliary(x1, x2, x3, x4)
        t1_out, t2_out, t3_out, t4_out = self.imtm(t1, t2, t3, t4)
        f1 = x1 + t1_out
        f2 = x2 + t2_out
        f3 = x3 + t3_out
        f4 = x4 + t4_out
        out = self.pool(f4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Initialize model and load weights
device = torch.device("cpu")
model = IdentityMappingResFormer(num_classes=4).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Class names
class_names = ['COVID', 'Viral Pneumonia', 'Lung_Opacity', 'Normal']

# Image preprocessing
test_transforms = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Apply transformations
    transformed = test_transforms(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_image(image):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        image_tensor = preprocess_image(image)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        all_probs = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
        
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': all_probs
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Read and process the image
            image = Image.open(file.stream)
            
            # Make prediction
            result = predict_image(image)
            
            # Convert image to base64 for display
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'prediction': result,
                'image': img_str
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)