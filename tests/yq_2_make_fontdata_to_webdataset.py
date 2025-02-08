import os
import tarfile
from tqdm import tqdm
import math 
import json
import torch
import random

from open_clip.tokenizer import FontTokenizer
from torch.utils.data import WeightedRandomSampler

import webdataset as wds
from webdataset.pytorch import IterableDataset
import torchvision.transforms as th_tfms
from functools import partial
from PIL import Image
import numpy as np
import time
from torch.utils.data import DataLoader
import json
from open_clip_train.data import SharedEpoch,DataInfo

def local_file_to_tar(rootpath:str,
                      tar_savepath:str,
                      tar_filesize:int,
                      tokenizer:FontTokenizer,
                      static_json:str,
                      epochs:int=1,
                      debug:bool=False,
                      ):
    
    # Load Static_json
    with open(static_json, "r", encoding="utf-8") as j:
        statis = json.load(j)

    content_stat = statis["content_stat"]

    files_to_tar_img, files_to_tar_caption, sample_weight = [], [], []
    rootpath = [rootpath] if "," not in rootpath else rootpath.split(",")

    style_paths = []
    for rp in rootpath :
        style_paths += [os.path.join(rp, s) for s in os.listdir(rp) if os.path.isdir(os.path.join(rp, s))] # rp1/s1, rp2/s2
        
    style_paths_sorted = sorted(style_paths, key=lambda x:x.split("/")[-1])
    style_names = [ x.split("/")[-1] for x in style_paths_sorted]
    num_style = len(style_names)
    print("==========================================")
    print("Total style : ", num_style)
    
    cont_map = tokenizer.json_data["ann_file"]
    for style_path in tqdm(style_paths_sorted, total=num_style, desc="Loading Style by Style"):
        subpath = style_path
        all_imgs = [f for f in os.listdir(subpath) if ".png" in f]

        for img_file in all_imgs:
            
            human_label = img_file.split("_")[1]
            stroke_line = cont_map[human_label]

            n_output_type = 4 if "<" in stroke_line else 3
            stroke_strs = [tokenizer.encode(stroke_line, output_type=i )["input_str"] for i in range(1, n_output_type+1)]
            stroke_strs = ["|".join(stroke_s) for stroke_s in stroke_strs ]                
            captions = "\t".join(stroke_strs)       # 硼	硼|<|##!L|石|(|#j|#s|#f|#c|#j|)|##!R|朋|(|#s|#r|#j|#j|#s|#r|#j|#j|)|>	硼|<|##!L|石|(|)|##!R|朋|(|)|>	<|##!L|石|(|)|##!R|朋|(|)|>

            files_to_tar_img.append(os.path.join(subpath, img_file))    # write image path
            files_to_tar_caption.append(captions)                       # write caption info
            sample_weight.append( 1/ content_stat[human_label])
            
    tensor_w = torch.as_tensor(sample_weight,dtype=torch.double)
    sampler = WeightedRandomSampler(tensor_w, len(files_to_tar_img)*epochs, replacement=True)

    if debug:
        total_num = len(list(sampler))
        statis_info = {}
        for s in tqdm(sampler, total=total_num, desc="statis contents info"):
            img_file = files_to_tar_img[s]
            human_label = img_file.split("/")[-1].split("_")[1]
            if human_label not in statis_info:
                statis_info[human_label] = 1
            else:
                statis_info[human_label] += 1
        
        content_keys = statis_info.keys()
        for k in content_keys:
            statis_info[k] = "%d-(%.4f %%)"%(statis_info[k], statis_info[k]*100/total_num)

        with open("tests/yq_2_statis_info.json", "w", encoding="utf-8") as f:
            json.dump(statis_info, f, indent=4,ensure_ascii=False)

        return 


    
    # Save to tar_file
    n_tar = math.ceil(len(files_to_tar_img)*epochs / tar_filesize) 
    print("Make Tar :", n_tar)
    tar_filepath = os.path.join(tar_savepath, f"fontV4-%04d.tar")
    global_i = 0
    with wds.ShardWriter(tar_filepath, maxcount=tar_filesize) as f:
        for idx in tqdm(sampler, desc="Save to Tar"):
            f_img = files_to_tar_img[idx]
            f_txt = files_to_tar_caption[idx]

            with open(f_img, "rb") as stream:
                image = stream.read()
            caption = f_txt.encode()    # utf-8 encode

            img_arc = "/".join( f_img.split("/")[-2:])
                
            data_ele = {"__key__":str(global_i)+"_"+img_arc, 
                        "jpg":image,    # in tar this key is "png.png" ,the value is bytes
                        "txt":caption   # in tar this key is "png.txt", the value is bytes
                        }
            f.write(data_ele)

            global_i += 1



    print("Save Done.")

def extract_tar(tar_filepath:str, extract_path:str="./"):
    with tarfile.open(tar_filepath, "r") as tar:
        print("Tar file contents:")
        tar.list()

        # Extract file to dir
        tar.extractall(extract_path)
        print("Extracted all files to :", extract_path)


def load_webdataset(url:str):

    dataset = wds.WebDataset(url)
    bar = tqdm(dataset)

    data_list = []
    result = {}
    for data in bar:
        bar.set_description_str("tar_name:"+data["__url__"].split("/")[-1])
        
        human_label = data["__key__"].split("/")[-1].split("_")[1]
        if human_label in result:
            result[human_label] += 1
        else:
            result[human_label] = 1
        data_list.append(data["__key__"])

        # image = data["png.jpg"]
        # txt = data["png.txt"].decode("utf-8")
    return result

######################################################
# time : 2024-12-09
# 1) build a list
# 2) using wds.webdata.shardlist.SimpleShardList() to wrap it
######################################################
def preprocess(sample, 
               tokenizer:FontTokenizer=None,
               tfms:th_tfms.Compose=None,
              
               ):
    # first in dataset , second in preprocess
    # sample is list or tuple
    # Return must be [image tensor, input_ids tensor]
    imgpath, slabel = sample

    # get image data
    with Image.open(imgpath) as image:
        image_np = np.asarray(image)
        if image_np.ndim == 3:
            image_np = image_np[:, :, 1]    # get render image data from green channel in V4 Dataset
        image_3c = np.expand_dims(image_np, axis=-1)
        image_3c = np.repeat(image_3c, 3, axis=-1)

        image = Image.fromarray(image_3c, mode="RGB")   # convert np.ndarray to PIL.Image   
    
    images = tfms(image)

    # get text data
    human_label = imgpath.split("/")[-1].split("_")[1]
    cont_map = tokenizer.json_data["ann_file"]
    stroke_line = cont_map[human_label]
    n_output_type = 4 if "<" in stroke_line and "<"!=stroke_line else 3
    stroke_strs = [tokenizer.encode(stroke_line, output_type=i)["input_ids"] for i in range(1, n_output_type+1)]
    input_ids = random.choice(stroke_strs)
    texts = torch.LongTensor(input_ids)

    return images, texts


def get_font_names(rootpath:str, ):
    rootpath = [rootpath] if "," not in rootpath else rootpath.split(",")
    base = []
    
    style_paths = []
    for rp in rootpath :
        style_paths += [os.path.join(rp, s) for s in os.listdir(rp) if os.path.isdir(os.path.join(rp, s))] # rp1/s1, rp2/s2
    
    
    style_paths_sorted = sorted(style_paths, key=lambda x:x.split("/")[-1])[:5]
    for slabel, style in enumerate(style_paths_sorted) :
        imgfiles = os.listdir(style)[:500]
        for img_f in tqdm(imgfiles, total=len(imgfiles), desc="style :"+style.split("/")[-1]):
            base.append({"png":os.path.join(style, img_f), "slabel":slabel})

    return base

#Copied from :https://github.com/webdataset/webdataset/blob/1e47aa3de66a23fed0d0f28e787c46beb98ac538/webdataset/shardlists.py#L169
class SimpleShardList_Font(IterableDataset):

    """An iterable dataset yielding a list of URLs."""

    def __init__(self, urls, seed=None, statis_json=None, epoch=-1):
        """Initialize the SimpleShardList.

        Args:
            urls (str or List[str]): A list of URLs as a Python list or brace notation string.
            seed (int or bool or None): Random seed for shuffling; if None, no shuffling is done,
                if True, a random seed is generated.
        """
        super().__init__()
        
        self.urls = urls
        if seed is True:
            seed = time.time()
        self.seed = seed
        
        if statis_json :
            assert os.path.isfile(statis_json), "%s not exist!"%statis_json
            with open(statis_json, "r" , encoding="utf-8") as j :
                statis = json.load(j)
            self.content_stat = statis["content_stat"]
            self.weight = [1/self.content_stat[url["png"].split("/")[-1].split("_")[1]] for url in self.urls]
        else:
            self.weight = statis_json

        self.epoch = epoch

    def __len__(self):
        """Return the number of URLs in the list.

        Returns:
            int: The number of URLs.
        """
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """

        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        urls = self.urls
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)

        if self.weight:
            g = torch.Generator()
            g.manual_seed(epoch)
            # Resample with weight
            rnd_indexes = WeightedRandomSampler(self.weight, num_samples=len(urls), 
                                                replacement=True, generator=g)
            url_reindex = [urls[r_i] for r_i in rnd_indexes]
        else:
            url_reindex = url

        for  url in url_reindex:
            yield url

def test_wdswarp(rootpath, tokenizer, tfms):

    base = get_font_names(rootpath)

    shared_epoch =  SharedEpoch(epoch=101)

    dataset = wds.DataPipeline(
        SimpleShardList_Font(base, 
                             statis_json="data/font_statis_1712.json", 
                             epoch=shared_epoch),
        wds.to_tuple("png", "slabel"),
        wds.split_by_node,
        wds.split_by_worker,
        wds.map(partial(preprocess, tokenizer=tokenizer, tfms=tfms)),
        wds.batched(3)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        persistent_workers=True
    )

    datainfo = DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

    for epoch in tqdm(range(3)):

        datainfo.set_epoch(epoch)
        dl = datainfo.dataloader
        for d in tqdm(dl):
            batch2 = d

            images2,texts2 = batch2
        pass


if __name__ == "__main__":

    tokenizer = FontTokenizer("data/font_vocab_1712.json", 77)
    rootpath = "/home/yue/Data/Data_v4/Data_v4_hand_c2W_latin_sz256,/home/yue/Data/Data_v4/Data_v4_design_c2W_latin_sz256"
    tar_savepath = "data/webdataset"
    tar_filesize = 250000 # 1000000 1w~141M,  25w~3.5G

    size = 224
    tfms = th_tfms.Compose([
        th_tfms.Resize(size),
        th_tfms.ToTensor(),
        th_tfms.Normalize(mean=[0.5,]*3, std=[0.5,]*3)
    ])

    # 1. Make Tar Format Dataset 
    # local_file_to_tar(
    #     rootpath=rootpath,
    #     tar_savepath=tar_savepath,
    #     tar_filesize=tar_filesize,
    #     tokenizer=tokenizer,
    #     static_json="data/font_statis_1712.json",
    #     epochs=20,debug=True
    # )


    # 2.Build Webdataset to load tar file
    # load_webdataset(os.path.join(tar_savepath, "fontV4-{0000..0004}.tar"))

            
    # 3. IterData warp 
    test_wdswarp(rootpath, tokenizer, tfms)




