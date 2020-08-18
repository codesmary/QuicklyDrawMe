import torch
from torchvision import models
from torch import cuda
from PIL import Image
import numpy as np

def load_checkpoint(path):
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def predict(image_path, model):
    img_tensor = process_image(image_path)

    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)

        top_class = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ][0]

        return top_class

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    img = image.resize((256, 256))

    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    img = np.array(img).transpose((2, 0, 1)) / 256

    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor

checkpoint_path = 'vgg16-transfer-4.pth'
train_on_gpu = cuda.is_available()

if train_on_gpu:
    gpu_count = cuda.device_count()
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

loaded_model, optimizer = load_checkpoint(path=checkpoint_path)

print(predict("pictures/test.jpg", loaded_model))