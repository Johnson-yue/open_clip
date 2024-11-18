#########################################################################
# This is naive function
# 1 font about 3-4 minute, maybe better later
# time : 2024-10-09
#########################################################################


import torch
from PIL import Image
import open_clip
import torchvision.transforms as th_tfms
import os
from tqdm import tqdm
from typing import Union
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
from src.font_tools import Font_Generator_v4


# CLIP Inference
class FontClip_Inference():

    def __init__(self, device=None, size=224, text_bs=60):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "logs/CLIP_font_ViT_L14_s1712_aug/checkpoints/epoch_8.pt"
        model_name = "ViT-L-14"
        ann_file = "data/font_strokes_v2_lv1_release.txt"
        token_dict="data/stroke_token_dict.txt"


        self.model  = open_clip.create_model(
            model_name, 
            pretrained=model_path,
            precision="amp",
            device=self.device
            ).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name, context_length=77, token_dict=token_dict)

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

            self.captions += [self.tokenizer([stroke_s]).unsqueeze(0) for stroke_s in stroke_strs]

        self.caption_token = torch.cat(self.captions, dim=0).to(self.device)
        cap_chunks = self.caption_token.size(0) // text_bs

        self.caption_token_chunks = self.caption_token.chunk(chunks=cap_chunks, dim=0)
        self.text_features = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for cap in tqdm(self.caption_token_chunks):
                text_features = self.model.encode_text(cap)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                self.text_features.append(text_features)

        self.text_features = torch.cat(self.text_features, dim=0)

    def __call__(self, img:list[Image.Image, str]):
        err_out = ""
        if isinstance(img, str):
            if os.path.isfile(img):
                img = Image.open(img)
            else:
                err_out += f"{img} not found\n"
        
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
        
        # processor image
        image = self.img_preprocess(img).unsqueeze(0).to(self.device)

        # encode image
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features =  self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # compute similarity
        similarity = (100.0 *image_features @ self.text_features.T).softmax(dim=-1)
        

        # get top 10 similarity
        top_values, top_indices = similarity.topk(10, dim=-1)

        output = "Result: \n"
        output += "score: \t | string: \n"
        output += "-----------------------\n"
        for i, (value, index) in enumerate(zip(top_values[0], top_indices[0])):
            output += f"top{i+1}: {value:.4f} - {self.caption_str[index]} \n"

        out = {"str": output, "top_values": top_values.cpu(), 
               "top_indices": top_indices.cpu(), "err":err_out,
               "label":self.caption_str, "human_label":self.human_label}

        return out


def check_ttf(ttf_file:Union[str, list],
              human_label:list[str],
              ):
    
    if isinstance(ttf_file, str):
        ttf_file = [ttf_file, ]
    
    # Render ttf image
    g = Font_Generator_v4(human_label, font_size=256)

    # CLIP Inference
    clip = FontClip_Inference()
    
    top1, top5, top10 , total_n = 0,0,0,0
    wrong_info = []

    for ttf in ttf_file:

        render_result = g.get_plain(ttf)
        for style_name, style_dict in render_result.items():
            np_arrs = style_dict["renderImg"][0]
            err_list = style_dict["err"]

            for c, arr in tqdm(np_arrs):
                pil_in = Image.fromarray(arr[:,:, 1]).convert("RGB")
                clip_out = clip(pil_in)

                
                top_values, top_indices = clip_out["top_values"][0], clip_out["top_indices"][0]
                cap_label, template_label = clip_out["label"], clip_out["human_label"]

                pred_c_list = [template_label[top_id] for top_id in top_indices] 
                pred_v = top_values[0]
                if c in pred_c_list[:1]:
                    top1 += 1
                    top5 += 1
                    top10 += 1
                elif c in pred_c_list[:5]:
                    wrong_info += [f"not in top1:{style_name}-{c} - pred:{template_label[top_indices[0]]}--{cap_label[top_indices[0]]}\n"]
                    top5 += 1
                    top10 += 1
                elif c in pred_c_list[:10]:
                    top10 += 1
                    wrong_info += [f"not in top5{style_name}-{c} - pred:{template_label[top_indices[0]]}--{cap_label[top_indices[0]]}\n"]
                else:
                    wrong_info += [f"not in top10{style_name}-{c} - pred:{template_label[top_indices[0]]}--{cap_label[top_indices[0]]}\n"]
                
                total_n += 1
   
    print("Check Done")
    
    result = {"top1": top1/total_n, 
              "top5": top5/total_n,
              "top10": top10/total_n,
              "wrong_info": wrong_info}
    return result



if __name__ == "__main__":

    ttf_file = "/home/yue/DataSets/Font_Data_v4/HandV4_1018/少女的祈祷.ttf"

    hand_charset = "./data/font_strokes_v2_lv1_release.txt"
    with open(hand_charset, "r") as f:
        charset = [line.strip("\n")[0] for line in f.readlines()][:20000]
    
    result = check_ttf(ttf_file, charset)
    print(result)