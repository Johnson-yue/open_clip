import torch
from PIL import Image
import open_clip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path =  '/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/OpenAI_CLIP/ViT-L-14.pt' # 'ViT-L/14'
model_name = "ViT-L-14"

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

img_path = "docs/CLIP.png"

image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text = tokenizer(["a diagram", "a dog", "a cat"]).cuda(device=device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 *image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", similarity)
# print(similarity)
