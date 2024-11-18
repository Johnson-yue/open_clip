import os, torch
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms as th_tfms
from torchvision.transforms import InterpolationMode 
from PIL import Image
import random, re, time
import numpy as np
import concurrent.futures as futures
import csv

# PartInfo
pair_part ={
    "!L":"!R",  "!R": "!L",
    "$U":"$D", "$D":"$U" ,
    "&LU":"&RD", "&RD":"&LU",
    "&LD":"&RU", "&RU":"&LD",
    "*":"*"
}
specifical_part = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅳ", "Ⅴ", "Ⅵ", 
                   "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ",
                   "ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ","ⅵ","ⅶ",
                   "ⅷ","ⅸ", "ⅹ",
                   "0","1","2","3","4","5","6","7","8","9", 
                   ",", ".", "-", "_", ":", "‰",
                   ] + [s for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]


class FontStrokenizer():

    def __init__(self,token_dict, max_content_len=77,
                 only_part=False, has_cont=False, **kwargs):
        
        # load token dict
        with open(token_dict, "r") as f:
            token_dict = [line.strip("\n") for line in f.readlines()]

        # Convert list-O(n) to dict-O(1)
        self.token_dict = {ele:i for i , ele in enumerate(token_dict)}
        self.len = len(self.token_dict)
        # reverse k-v pair from ids to token_str
        token_reverse = {str("%05d"%v):k for k,v in self.token_dict.items()}
        self.token_dict.update(token_reverse)

        self.only_part = only_part
        self.has_cont = has_cont
        self.max_length = max_content_len

        print("=="*20)
        print(f"token dict length:{self.len}, total len:{len(self.token_dict)}")
        print("max annotation length:", self.max_length)
        print("=="*20)

    def get_p1p2(self, ann_line):
        """ from ann_line to part1+part2
        Input:
            丕<$U不(jsfk)$D一(j)>
        Output:
            丕/##$U/不/jsfk/##$D/一/j/
        """
        assert "<" in ann_line, "ann:[%s] error, have no '<'"%ann_line
        
        # get strokes (p1_strokes, p2_strokes)
        strokes = re.finditer(r'[a-z]+', ann_line)

        # get position encode (p1_pos, p2_pos)
        matches = re.finditer(r"(!L|!R|\$U|\$D|&LU|&LD|&RU|&RD|\*)", ann_line)

        c = ann_line[0]
        out = (c,)  # add context to output
        for match, stroke in zip(matches, strokes):
            part_pos = match.group()
            pos_end_idx = match.end()

            part_c = ann_line[pos_end_idx]
            if part_c in specifical_part:
                part_c = "#"+part_c

            part_stroke = stroke.group()
            out += ("##"+part_pos, part_c, part_stroke)     # add "##" to part_pos
        return out


    def encode(self, ann_line:str, output_type:int=1):
        
        if "<" not in ann_line or "<" == ann_line:
            # get unsplitable content
            
            if output_type == 1:                 # single char : [亭]
                input_str = [ann_line[0]]
            elif output_type == 2:               # char and stroke : [亭 ( #k #j #f #c #j #d #e #j #g )] 
                input_str = [ann_line[0], "(",] + ["#"+l for l in ann_line[2:-1]] +[")"] 
            elif output_type == 3:
                input_str = [ann_line[0], "(",")" ]
            else:
                input_str = [ann_line[0]]       # [亭()]
        else:
            # get splitable content
            stroke_info = self.get_p1p2(ann_line)

            if output_type == 1:                       # single char : [交]
                input_str = [ann_line[0]]
            elif output_type == 2:                     # char and stroke : [交 < ##$U 亠 ( #k #j ) ##$D 父 ( #s #k #s #l ) > ]
                input_str = [stroke_info[0], "<", stroke_info[1], stroke_info[2], "("]
                input_str += ["#"+s for s in stroke_info[3]] + [")"]  
                input_str += [stroke_info[4], stroke_info[5]] + ["("] 
                input_str += ["#"+s for s in stroke_info[6]] + [")", ">"] 
            elif output_type == 3:                     # only part : [交 < ##$U 亠 ( ) ##$D 父 (  ) > ]
                input_str = [stroke_info[0], "<",stroke_info[1], stroke_info[2], "(", ")", 
                                stroke_info[4], stroke_info[5], "(", ")",  ">"]
            elif output_type == 4:                     # only part : [< ##$U 亠 ( ) ##$D 父 (  ) > ]
                input_str = ["<",stroke_info[1], stroke_info[2], "(", ")", 
                                stroke_info[4], stroke_info[5], "(", ")",  ">"]
            else:
                input_str = input_str

        num_in = len(input_str)
        assert num_in <= self.max_length, f"input_str len:{num_in} > max_length:{self.max_length}"
        mask = [True]*num_in

        # pad to max_length with [pad] token
        input_ids = [self.token_dict[s] for s in input_str] + \
                    [self.token_dict["pad"], ]*(self.max_length - num_in)
        mask_padded = mask + [False] *(self.max_length - num_in)

        out = {"input_ids":input_ids, "mask":mask_padded, "length":num_in, 
               "input_str":input_str}
        return out   
    
    def get_embedding(self, s:str):
        # s must in token_dict
        if s in pair_part:
            s = "##"+s
        
        return self.token_dict[s]
    
    def get_UncondEmbedding(self):
        """Return Un-Conditional Token Ids"""
        return {
            "content":self.token_dict["UnCont"],
        }
    
    def decode(self, encode_info, rm_pad=True):
        out = []   
        if isinstance(encode_info, dict):
            input_ids = encode_info["input_ids"]
            num_in = encode_info["length"]

            input_ids_nopad = input_ids[:num_in]
            str_list = [self.token_dict["%05d"%ii] for ii in input_ids_nopad]
            if rm_pad :str_list = ["" if s=="pad" else s for s in str_list]
            out = "|".join(str_list)

        elif isinstance(encode_info, list):
            
            for enc_info in encode_info:
                if isinstance(enc_info, list):
                    ele = [self.get_embedding("%05d"%e_info) for e_info in enc_info]
                    if rm_pad :ele = ["" if e =="pad" else e for e in ele]
                    ele = "".join(ele) +"\n"
                else:
                    ele = self.get_embedding("%05d"%enc_info)
                    if rm_pad and  ele == "pad": ele = ""
                out.append(ele)
            
            out = "|".join(out)
        return out
    
    def __call__(self, ann_line:list[str], **args):
        if isinstance(ann_line, str):
            ann_line = [ann_line]
        
        out = []
        for ann in ann_line:
            output_type = 1 if "output_type" not in args else args["output_type"]
            out.append(self.encode(ann, output_type)["input_str"])

        if len(ann_line) == 1:
            out = out[0]
        return out
        

    

# Multi_processor_read_data
def read_information_from_folder(folder_path:str, tokenizer:FontStrokenizer, phase:str, val_num:int,
                                cont_map:dict, seed:int):
    """ get img_path,  statistics_info"""
    folder_info = []
    folder_dict = {"img_captions":[],  
                    }

    all_imgfiles = os.listdir(folder_path)
    random.seed(seed)
    random.shuffle(all_imgfiles)
    #-----------------param:val_num ------------------
    if val_num != 0:
        imgfiles = all_imgfiles[:-val_num] if phase.find("train") != -1 else all_imgfiles[-val_num:]
    else:
        imgfiles = all_imgfiles

    imgfiles = sorted(imgfiles)

    # ----------------- read every image file --------------
    for imgf in imgfiles:
        imgpath = os.path.join(folder_path, imgf)

        if os.path.isfile(imgpath):

            _sampler = [imgpath]
            
            # statistic content and style number
            c = imgpath.split("/")[-1].split("_")[1]
            
            # add part and stroke into stat
            if tokenizer is not None:
                stroke_line = cont_map[c]
                n_output_type = 4 if "<" in stroke_line else 3
                stroke_strs = [tokenizer.encode(stroke_line, output_type=i )["input_str"] for i in range(1, n_output_type+1)]
                stroke_strs = ["|".join(stroke_s) for stroke_s in stroke_strs ]                
                captions = "\t".join(stroke_strs)

                _sampler.append(captions)


            folder_dict["img_captions"].append(_sampler)

    folder_info.append(folder_dict)

    return folder_info

class HandFont_DataMaker():

    def __call__(self,
                 rootpath,
                 ann_file=None,
                 phase="train",
                 val_num=0,
                 seed=20240905,
                 **args):
        
        self.base = []
        self.is_train = "train" in phase

        # ----------param: rootpath-----------------------
        # load image file names
        rootpath = [rootpath] if "," not in rootpath else rootpath.split(",")

        style_paths = []
        for rp in rootpath :
            style_paths += [os.path.join(rp, s) for s in os.listdir(rp) if os.path.isdir(os.path.join(rp, s))] # rp1/s1, rp2/s2
        
        style_paths_sorted = sorted(style_paths, key=lambda x:x.split("/")[-1]) 
        self.styles = [ x.split("/")[-1] for x in style_paths_sorted]
        num_style = len(self.styles)
        print("Total style : ", num_style)
    
        # --------- param: ann_file ---------------------
        if ann_file is not None:
            with open(ann_file, "r") as f:
                lines = [line.strip("\n") for line in f.readlines()]
            self.cont_map = {line[0]:line for line in lines}

    
        # -------- param:phase --------------------
        if "train" in phase:
            if "-" in phase:
                try:
                    train_num = int(phase.split("-")[-1])
                except:
                    train_num = num_style
                    print("train-all (%d) styles"%num_style)
            else:
                train_num = num_style
            styles = style_paths_sorted[:train_num]
        elif "debug" in phase:
            styles = style_paths_sorted[:3]
        elif "val" in phase :
            val_num_style = -20
            if "-" in phase:
                try:
                    val_num_style = int(phase.split("-")[-1])
                except:
                    val_num_style = num_style
                    print("val-all (%d) styles"%num_style)
            else:
                val_num_style = num_style
            styles = style_paths_sorted[:val_num_style]
        print("Training style number :", len(styles))

        # -------- param:seed --------------------
        random.seed(seed)
        print("Seed:", seed)
        rnd_style_seed = [random.randint(0, 999999) for i in range(len(styles))]

        # using multi-processor read folder collection image and caption from image file
        cpu_count = os.cpu_count()
        max_workers = min(cpu_count, len(styles))
        print("max_workers:", max_workers)
        all_folder_info = [] #["filepath", "title"]

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_folder = {executor.submit(read_information_from_folder,
                                                style, self.tokenizer, phase, val_num, 
                                                self.cont_map, rnd_style_seed[si]):  (si,style) for si, style in enumerate(styles)}
            for future in tqdm(futures.as_completed(future_to_folder), total=len(styles), desc="Load Data Folders"):
                folder = future_to_folder[future]
                try:
                    folder_info = future.result()
                    all_folder_info.extend(folder_info)
                except Exception as e:
                    print(f"Err in Folder: {folder}, err:{e}")
        print(f"Read Data Folder Finished Number: {len(styles)} .")

        # parser all_folder_info to self.base
        self.base= [["filepath", "title"],]
        for f_info in all_folder_info:
            self.base.extend(f_info["img_captions"])

        print("=="*20)

        return self.base

    def set_tokenizer(self, token_dict, max_content_len=77, **args):
        
        if token_dict is not None:
            self.tokenizer = FontStrokenizer(token_dict, max_content_len)
        else:
            self.tokenizer = token_dict
    

if __name__ == "__main__":

    data_dirs = [
                "Data_v4_hand_c2W_latin_sz256",  
                "Data_v4_design_c2W_latin_sz256",
                #  "Data_v4_hand_stroke_better1_c2W_latin_sz256",
                #  "Data_v4_design_stroke_better1_c2W_latin_sz256",
                #   "Test_tokenizer",    # Test-Strokenizer,
                #   "Test_tokenizer_stroke", # Test-Strokenizer with Stroke info.
                 ]
    rootpath = ",".join([os.path.join("/home/yue/Data/Data_v4", d) for d in data_dirs])

    token_dict = "data/stroke_token_dict.txt"
    ann_file = "data/font_strokes_v2_lv1_release.txt"
    val_num = 10
    num_font = "5"    # "5"
    csv_train_file = "data/csv/train_fontV4_tiny.csv"
    csv_val_file = "data/csv/val_fontV4_tiny.csv"

    DataMaker = HandFont_DataMaker()
    DataMaker.set_tokenizer(token_dict, 77)

    # write val_data to CSV
    val_data  = DataMaker(rootpath, ann_file, "val-%s"%num_font, val_num)
    with open(csv_val_file, "w", newline='', encoding="utf-8") as csvfile_val:
        writer = csv.writer(csvfile_val)
        writer.writerows(val_data)
    print("Make val-data to csv file done.")

    # write train_data to CSV
    train_data = DataMaker(rootpath, ann_file, "train-%s"%num_font, val_num)
    with open(csv_train_file, "w", newline='', encoding="utf-8") as csvfile_train:
        writer = csv.writer(csvfile_train)
        writer.writerows(train_data)

    print("Make train-data to csv file done.")



    