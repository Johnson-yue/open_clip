import torch
from PIL import Image
import open_clip
import torchvision.transforms as th_tfms
import os
from tqdm import tqdm
import gradio as gr
from functools import partial
import sys
sys.path.append("/home/yue/DeepLearning/VISION_TEXT/open_clip")
from src.open_clip.tokenizer import FontTokenizer

class FontClip_Inference():

    def __init__(self, device=None, size=224):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "logs/CLIP_font_ViT_L14_s1712_aug/checkpoints/epoch_8.pt"
        model_name = "ViT-L-14"
        ann_file = "data/font_strokes_v2_lv1_release.txt"
        token_dict="data/stroke_token_dict.txt"
        self.name = "L14"

        self.model  = open_clip.create_model(
            model_name, 
            pretrained=model_path,
            precision="amp",
            device=self.device,
            load_weights_only=False,
            ).eval()
        self.tokenizer = FontTokenizer(token_dict=token_dict,context_length=77)

        self.img_preprocess = th_tfms.Compose([
            th_tfms.Resize(size), th_tfms.ToTensor(), th_tfms.Normalize(mean=[0.5,]*3, std=[0.5,]*3)
        ])

        with open(ann_file, 'r') as f:
            lines = [line.strip("\n") for line in f.readlines()]
        self.cont_map = {line[0]:line for line in lines}
        self.ann_lines = list(self.cont_map.values())

        self.captions, self.caption_str, self.human_label = [], [], []
        for cn_char, stroke_line in self.cont_map.items():
            n_output_type = 4 if "<" in stroke_line else 3
            stroke_strs = [self.tokenizer.encode(stroke_line, output_type=i)["input_str"] for i in range(1, n_output_type+1)]
            self.caption_str += stroke_strs
            self.human_label += [cn_char, ]*n_output_type
            stroke_strs = ["|".join(stroke_s) for stroke_s in stroke_strs ]  

            self.captions += [self.tokenizer([stroke_s]) for stroke_s in stroke_strs]

        self.caption_token = torch.cat(self.captions, dim=0).to(self.device)
        cap_chunks = self.caption_token.size(0) // 60

        self.caption_token_chunks = self.caption_token.chunk(chunks=cap_chunks, dim=0)
        self.text_features = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for cap in tqdm(self.caption_token_chunks):
                text_features = self.model.encode_text(cap)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                self.text_features.append(text_features)

        self.text_features = torch.cat(self.text_features, dim=0)



    def __call__(self, img:list[Image.Image, str], txt=""):
        err_out = ""
        
        if isinstance(img, str):
            if os.path.isfile(img):
                img = Image.open(img)
            else:
                err_out += f"{img} not found\n"
        
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
        
        # processor txt
        if txt != "":
            new_add = []
            new_add_human_label = []

            txts = [t.split("--")[-1] for t in  txt.split("\n") if t!=""]
            txt = [t[0]+t[1:].replace(" ", "")  for t in txts]
            txt_input_ids = self.tokenizer(txt)

            for t in txt:
                t_sample = [ti for ti in t.replace("|", "").replace("##", "").replace("#", "")]
                new_add.append(t_sample)
                new_add_human_label.append(t.replace("|", "").replace("##", "").replace("#", ""))

            new_caption_str = self.caption_str + new_add
            new_human_label = self.human_label + new_add_human_label

            txt_input_ids = txt_input_ids.to(device=self.text_features.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = self.model.encode_text(txt_input_ids)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                all_text_features = torch.cat([self.text_features, text_features], 0)
        
        else:
            new_caption_str = self.caption_str
            new_human_label = self.human_label


        # processor image
        image = self.img_preprocess(img).unsqueeze(0).to(self.device)

        # encode image
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features =  self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # compute similarity
        all_text_features = self.text_features if txt == "" else all_text_features
        similarity = (100.0 *image_features @ all_text_features.T).softmax(dim=-1)
        

        # get top 10 similarity
        top_values, top_indices = similarity.topk(10, dim=-1)

        output = "Result: \n"
        output += "score: \t | string: \n"
        output += "-----------------------\n"
        for i, (value, index) in enumerate(zip(top_values[0], top_indices[0])):
            output += f"top{i+1}: {value:.4f} - {new_caption_str[index]} \n"

        out = {"str": output, "top_values": top_values.cpu(), 
               "top_indices": top_indices.cpu(), "err":err_out,
               "label":new_caption_str, "human_label":new_human_label}

        return out
    

class FontClip_Inference_H14(FontClip_Inference):

    def __init__(self, device=None, size=224,
                 model_name="ViT-H-14",
                 model_path="logs/CLIP_font_ViT_H14_s1712/checkpoints/epoch_3.pt",
                 token_dict="data/font_vocab_1712.json",
                 ann_file="") :
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "H14"

        self.model = open_clip.create_model(
            model_name=model_name,
            pretrained=model_path,
            precision="amp",
            device=self.device,
            load_weights_only=False
        ).eval()

        self.tokenizer = FontTokenizer(token_dict=token_dict, context_length=77)
        self.img_preprocess = th_tfms.Compose([
            th_tfms.Resize(size), th_tfms.ToTensor(), th_tfms.Normalize(mean=[0.5,]*3, std=[0.5,]*3)
        ])

        if token_dict.endswith("json"):
            self.cont_map = self.tokenizer.json_data["ann_file"]
        else:
            with open(ann_file, "r") as f:
                lines = [line.strip("\n") for line in f.readlines()]
            self.cont_map = {line[0]:line for line in lines}


        self.captions, self.caption_str, self.human_label = [], [], []
        for cn_char, stroke_line in self.cont_map.items():
            n_output_type = 4 if "<" in stroke_line else 3
            stroke_strs = [self.tokenizer.encode(stroke_line, output_type=i)["input_str"] for i in range(1, n_output_type+1)]
            self.caption_str += stroke_strs
            self.human_label += [cn_char, ]*n_output_type
            stroke_strs = ["|".join(stroke_s) for stroke_s in stroke_strs ]  

            self.captions += [self.tokenizer([stroke_s]) for stroke_s in stroke_strs]

        self.caption_token = torch.cat(self.captions, dim=0).to(self.device)
        cap_chunks = self.caption_token.size(0) // 60

        self.caption_token_chunks = self.caption_token.chunk(chunks=cap_chunks, dim=0)
        self.text_features = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for cap in tqdm(self.caption_token_chunks):
                text_features = self.model.encode_text(cap)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                self.text_features.append(text_features)

        self.text_features = torch.cat(self.text_features, dim=0)



def test_one_image():
    font_clip = FontClip_Inference()
    img = Image.open("/home/yue/Data/Data_v3/Data_v3_design_c2W_sz256/HYHeiMiTiJ/C10068_畔_256.png").convert("RGB")

    out = font_clip(img)
    print(out["str"])
    print(out["err"])

def gradio_func(img, newtxt, model):
    """
    Input: 
        img: PIL.Image
        newtxt: str, not common skeleton

    Output: str
    """

    if isinstance(img, Image.Image) is False:
        return {"no-img": 1.0}
    out = model(img, txt=newtxt)

    top_values, top_indices = out["top_values"][0], out["top_indices"][0]
    cap_label,human_label = out["label"], out["human_label"]

    result = {}
    for i , (value, index) in enumerate(zip(top_values, top_indices)):
        result[human_label[index]+"--"+"|".join(cap_label[index])] = float(value)

    return result,result


def gradio_ui(model):

    model_name = model.name

    with gr.Blocks() as demo:
        gr.Markdown("""
                    # Font Clip Demo (%s)
                    char_size : 2w
                    """%model_name)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                label_out_report =  gr.Label(label="Output", num_top_classes=10)
                skeleton_input = gr.Text(lines=10, placeholder="Add extra Prompt", label="Other Skeleton", show_label=True)
                
            with gr.Column():
                label_out =  gr.Label(label="Output", num_top_classes=10)
                btn = gr.Button("Predict")


        # Event
        image_input.change(partial(gradio_func, model=model),
                           inputs=[image_input, skeleton_input],
                           outputs=[label_out,label_out_report],
                           )
        btn.click(partial(gradio_func, model=model),
                           inputs=[image_input, skeleton_input],
                           outputs=[label_out,label_out_report],)
        
    return demo

if __name__ == "__main__": 

    model_name = "L14"  #["L14", "H14"]

    if model_name == "L14":
        font_clip = FontClip_Inference()        # CLIP-L-14
    elif model_name == "H14":
        font_clip = FontClip_Inference_H14()


    demo = gradio_ui(font_clip)

    demo.launch()