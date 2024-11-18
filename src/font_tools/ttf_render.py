from freetype import *
import numpy as np
import os, time, cv2
from tqdm import tqdm
from typing import Union

####################################################
# Copied from : /home/yue/DeepLearning/Diffusions/rewrite_diffusion_git/make_dataset
# time : 2024-10-09
# Make Data V4
####################################################
class Font_Generator_v4():

    def __init__(self, 
                 human_label: Union[str, list] = None, 
                 font_size: int = 256,
                 save_path:str="./font_data/"):
        
        # load generate character into list
        if human_label is None:
            human_label = []
        elif isinstance(human_label, str):
            with open(human_label, 'r') as f:
                human_label = [line.strip("\n") for line in f.readlines()]
        
        self.human_label = human_label
        self.font_size = font_size
        self.save_path = save_path
        self.save_dir = ""

    def get_glyph_metrics(self ,ttf,  face,  character, load_flag=FT_LOAD_RENDER):
        """获取一个glyph的 metric信息 包括：  只是一个例子，没有具体使用到
            xMin, yMin, yMax, Height, Width
        Input:
            ttf     -   *.ttf file
            face    -   freetype Face class
            character - "字"
            load_flag - 加载字符的选项 : 默认加载 ｜ 不加载位图信息
        Return :
            若成功，则返回
                {"xMin":xMin, "yMin":yMin, "yMax":yMax,
                "Height":Height, "Width":Width}
            若失败，则返回空字典
                {}
        """
        try:
            face.load_char(character, load_flag)
        except:
            print("Font: [%s]- face load char [%s] failed !!!"%(ttf, character))
            return {}
        
        glyph = face.glyph
        bitmap = face.glyph.bitmap

        width , rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
        top , left = glyph.bitmap_top, glyph.bitmap_left

        # 获取四个基本参数
        xMin = left
        xMax = left+width
        yMax = top
        yMin = rows - yMax
    
        # 获取bitmap
        Z2 = np.zeros((rows, width))
        dx, dy = 0,0
        data = np.array(bitmap.buffer[:rows*pitch]).reshape(rows, pitch)
        Z2[dx:dx+rows, dy:dy+width] = data[:, :width]

        # 输出
        return {"xMin": xMin, "xMax":xMax, "yMin": yMin, "yMax":yMax,
                "bitmap": Z2}

    def adaptive_fontSize(self, fontSize, face, char_list, bar_str=""):
        """ 不断缩小字号，直到满足下面所有的条件：
        1） 最右边 xMax 小于 fontSize
        2） 最上面 yMax 小于 fontSize
        3） 高 rows 小于fontSize
        4） 宽 width 小于 fontsize
        5） 最下面 yMin_i + 最上面 yMax_j 小于 fontSize  
        """

        fsize = fontSize

        # 获取单个字符的 rows ，width
        def get_char_size(face, char_c, fsize):
            face.set_pixel_sizes(0, fsize)

            try:
                char_ord = ord(char_c)
                face.load_char(char_ord, flags=FT_LOAD_RENDER)
                bitmap = face.glyph.bitmap
                glyph = face.glyph

                width, rows = bitmap.width, bitmap.rows
                top , left = glyph.bitmap_top, glyph.bitmap_left

                xMin, xMax = left, left+width
                yMax = top
                yMin = rows - yMax

                return xMin, xMax, yMin, yMax, rows, width
            except Exception as e:
                print("err:get_char_size -[%s]- "%bar_str, e)
                return  999, 0 ,999, 0, 0, 0


        # 先遍历一遍所有的字符，找到rows和width最高的字符
        min_xMin, max_xMax = [999, ""],[-999, ""]
        max_yMin, max_yMax = [-999, ""],[-999, ""]
        max_width, max_rows = [-999, ""],[-999, ""]

        for clabel, char_c in enumerate(char_list):
            
            try:
                if len(char_c) ==0:
                    print("[Err]: adaptive_fontSize:char_c(-%d-) length is 0"%char_c)
                    continue
                xMin, xMax, yMin, yMax, rows, width = get_char_size(face, char_c, fontSize)

                if rows ==0 or width == 0:
                    continue    # no this glyph

                if width > max_width[0]:
                    max_width[0] = width
                    max_width[1] = char_c
                if rows > max_rows[0]:
                    max_rows[0] = rows
                    max_rows[1] = char_c
                if xMin < min_xMin[0]:
                    min_xMin[0] = xMin
                    min_xMin[1] = char_c
                if xMax > max_xMax[0]:
                    max_xMax[0] = xMax
                    max_xMax[1] = char_c
                if yMin > max_yMin[0]:
                    max_yMin[0] = yMin
                    max_yMin[1] = char_c
                if yMax > max_yMax[0]:
                    max_yMax[0] = yMax
                    max_yMax[1] = char_c
            except Exception as e:
                print("[Err] adaptive font-size err-c(i=%d-%s):"%(clabel,char_c), e)
                continue

        # 判断：如果 最大的rows 或者 最大的 width 大于的 fontSize ，就缩小fontsize直到小于fontSize为止
        # 最后返回 fontsize
        xMax, yMin, yMax, rows, width, xMin = max_xMax[0], max_yMin[0], max_yMax[0], max_rows[0], max_width[0], min_xMin[0]
        xMax_c, yMin_c, yMax_c, rows_c, width_c, xMin_c = max_xMax[1],max_yMin[1], max_yMax[1], max_rows[1], max_width[1], min_xMin[1]
        if xMax_c == "" or yMin_c =="" or yMax_c == "" or rows_c == "" or width_c == "" or xMin_c == "":
            print("[Err]This Font have bug some Max-Value is None :[%s]"%bar_str)
            return 0 
        if rows ==0 or width == 0:
            # no latin glyph
            return 0 
        
        # Step 1: 根据最大字面的字符（width_c）确定 pad的大小
        xMin_width = get_char_size(face,width_c, fontSize )[0]
        yMax_rows = get_char_size(face, rows_c, fontSize)[3]
        col_pad = (fontSize - width) // 2   # col_pad must >= 0
        row_pad = (fontSize - rows) // 2    # row_pad must >= 0


        # Step 2: 根据最大字面的字符（width_c）和 pad 确定“初始基线（origin_col）” 和“边际条件-1”
        origin_col = col_pad - xMin_width
        origin_row = row_pad + yMax_rows
        cond_1_col = origin_col + xMin  #  cond_1_col 的边际条件是 >=0   
        cond_1_row = origin_row - yMax  #  cond_1_row 的边际条件是 >=0

        # cond_1 正常情况下应该大于等于0， 如果 小于0 ，那么就根据xMin来平移orgin_col
        new_origin_col = -xMin  # 令 cond_1_col == 0 ，则 new_origin_col = -xMin
        new_origin_row = yMax   # 令 cond_1_row == 0 ，则 new_origin_row = yMax

        # 根据新的基线(new_origin_col) 和 最大字面（width_c） 确定边际条件-2
        cond_2_col = new_origin_col + xMin_width + width  # cond_2_col 的边际条件, cond_2_col must <= fontsize
        cond_2_row = new_origin_row + yMin                # cond_2_row的边际条件,, cond_2_row must <= fontsize

        # xMax + 基线 也不能超过 fontSize
        cond_3_col = xMax + max(origin_col, new_origin_col)
        cond_3_row = yMin + max(origin_row, new_origin_row)

        while (xMax >= fontSize )or (yMax >=fontSize) or \
              (rows>=fontSize) or (width >= fontSize) or \
              ((yMin+yMax)>=fontSize) or (cond_1_col <0 and cond_2_col >= fontSize) or (cond_3_col >=fontSize) or \
               (cond_1_row <0 and cond_2_row >= fontSize) or (cond_3_row >=fontSize)     :
            fsize -= 1

            xMin = get_char_size(face, xMin_c, fsize)[0]
            xMax = get_char_size(face, xMax_c, fsize)[1]
            yMin = get_char_size(face, yMin_c, fsize)[2]
            yMax = get_char_size(face, yMax_c, fsize)[3]
            rows = get_char_size(face, rows_c, fsize)[-2]
            maxWidth = get_char_size(face, width_c, fsize)
            xMin_width, width = maxWidth[0], maxWidth[-1]
            yMax_rows = get_char_size(face, rows_c, fsize)[3]

            
            # Step 1
            col_pad = (fontSize - width) // 2   # pad 必须大于等于0
            row_pad = (fontSize - rows) // 2

            # Step 2:
            origin_col = col_pad - xMin_width   # 基线可以为 负
            origin_row = row_pad + yMax_rows
            cond_1_col = origin_col + xMin
            cond_1_row = origin_row - yMax

            # Step 3:
            new_origin_col = -xMin
            new_origin_row = yMax

            # Step 4:
            cond_2_col = new_origin_col + xMin_width + width 
            cond_2_row = new_origin_row  + yMin

            # Step 5:
            cond_3_col = xMax + max(origin_col, new_origin_col)
            cond_3_row = yMin + max(origin_row, new_origin_row)

            
        return fsize 
    
    def render_to_numpy(self, fit_fontsize, origin_fontsize, font_face, char_list, bar_str, debug=False):

        # 根据 【字体大小】 和 【字符列表】 找到绘制基线
        origin_point = self.find_origin(fit_fontsize, origin_fontsize, font_face, char_list, debug=debug)

        flags = FT_LOAD_RENDER
        # 保存图片到本地文件
        bar = tqdm(enumerate(char_list), total=len(char_list))
        bar.set_description_str(bar_str)
        err_list = []

        np_arrs = []
        for idxx, c in bar:
            
            if font_face.get_char_index(ord(c)) == 0 : # filter NOGlyPH
                    err_list.append(c)
                    continue
            
            try:
                font_face.load_char(ord(c), flags)
            except Exception as e:
                print("[Err] This char(%s) load from font Failed"%(c))
                err_list.append(c)
                bar.set_description_str(bar_str +"err_c:" +str(err_list))
                print(e)
                continue

            # get bitmap
            glyph = font_face.glyph
            bitmap = glyph.bitmap

            width , rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
            top , left = glyph.bitmap_top, glyph.bitmap_left

            if rows == 0 :
                err_list.append(c)
                print(" this char(%s) glyph row==0. skip it"%(c))
                continue
                
            # 获取bitmap
            Z2 = np.zeros((rows, width))
            dx, dy = 0,0 
            data = np.array(bitmap.buffer[:rows*pitch]).reshape(rows, pitch).astype(np.uint8)
            data = 255 - data       # 黑底白字 转 成白底黑字
            Z2[dx:dx+rows, dy:dy+width] = data[:, :width]

            draw_top = origin_point["origin_row"] - top
            draw_left = origin_point["origin_col"]+ left
            draw_img_g = np.ones((origin_fontsize, origin_fontsize), dtype=np.uint8)*255        # 空白部分使用255填充
            try:
                
                draw_rows = origin_fontsize - draw_top
                draw_cols = origin_fontsize - draw_left
                if draw_rows < rows:
                    Z2 = Z2[:draw_rows, :]
                if draw_cols < width:
                    Z2 = Z2[:, :draw_cols]

                draw_img_g[draw_top:draw_top+rows, draw_left:draw_left+width] = Z2      # 在 green 通道画字体数据

                # check only for debug : 画出基点坐标
                draw_img_r = np.ones_like(draw_img_g)*255
                draw_img_r[:, origin_point["origin_col"]] = 0     # 在 red 通道画 基线数据
                draw_img_r[origin_point["origin_row"], :] = 0

                draw_img_b = np.ones_like(draw_img_g)*255         # 在blue 通道画 left和 top的数据
                draw_img_b[:, draw_left] = 0
                draw_img_b[draw_top, :] = 0

                # concat rgb them
                draw_img = np.zeros((origin_fontsize, origin_fontsize, 3), dtype=np.uint8)
                draw_img[:,:,0] = draw_img_r
                draw_img[:,:,1] = draw_img_g
                draw_img[:,:,2] = draw_img_b

                np_arrs.append([c, draw_img])

            except Exception as e:
                err_list.append(c)
                bar.set_description_str(bar_str + "-sz_err:"+str(err_list))
                print(e)
                print(bar_str+ "[Err]info: c(%s)-top_left(%d,%d)-rw(%d,%d)"%(c , draw_top, draw_left, rows, width))
                continue
        
        return np_arrs, err_list

    def get_plain(self, ttf ):

        char_list = self.human_label
        char_size = [self.font_size]

        self.np_images = {}
        num_chars = len(char_list)
        ttf_name = ttf.split("/")[-1].split(".")[0]
        bar_str = "[%s] n=[%d]"%(ttf.split("/")[-1], num_chars)
        assert os.path.isfile(ttf), "[%s] not a file"%ttf
        save_path = os.path.join(self.save_path, ttf.split("/")[-1].split(".")[0])
        self.save_dir = save_path
        os.makedirs(save_path, exist_ok=True)

        face = Face(ttf)
        render_imgs = []
        for c_sz in char_size:
            
            font_size = c_sz
            #  绘制中文字符
            font_size = self.adaptive_fontSize(c_sz, face, char_list, bar_str=bar_str+"(ch)")
            if font_size !=0 :
                face.set_pixel_sizes(0, font_size)
                bar_str += "-SZ[%d]"%font_size
                np_arrs, err_list =  self.render_to_numpy(font_size, c_sz, face, char_list, bar_str=bar_str, debug=False)
                render_imgs.append(np_arrs)
            else:
                print(bar_str + "no chinese glyph")

        out = {"renderImg": render_imgs, "err":err_list}
        return {str(ttf_name):out}
            
    def find_origin(self, suit_size, font_size:int, ttf_face:Face, char_list:list, debug=False):
        """获取原点/基线的坐标， 一套字内所有原点位置应该是一致的，这样能保证重心一致"""
        
        ttf_face.set_pixel_sizes(0, suit_size)

        max_width, max_rows ={"val":-999, "char":"no_char"},{"val":-999, "char":"no_char"} 
        max_yMax, max_yMin = {"val":-999, "char":"no_char"}, {"val":-999, "char":"no_char"}
        min_xMin = {"val":999, "char":"no_char"}

        for ci, char_c in enumerate(char_list):
            if len(char_c):
                ord_c = ord(char_c) 
            else:
                print("[Err]: find_origin:char_c(-%d-) length is 0"%char_c)
                continue
            ttf_face.load_char(ord_c, flags=FT_LOAD_RENDER)

            glyph = ttf_face.glyph
            bitmap = glyph.bitmap

            width, rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
            top, left = glyph.bitmap_top, glyph.bitmap_left
            yMin = rows - top
            yMax = top
            xMin = left

            if width > max_width["val"]:
                max_width["val"] = width
                max_width["char"] = char_c
                max_width["left"] = left
            
            if yMin > max_yMin["val"]:
                max_yMin["val"] = yMin
                max_yMin["char"] = char_c
                max_yMin["rows"]  = rows
            
            if yMax > max_yMax["val"]:
                max_yMax["val"] = yMax
                max_yMax["char"] = char_c
                max_yMax["rows"]  = rows

            if xMin < min_xMin["val"]:
                min_xMin["val"] = xMin
                min_xMin["char"] = char_c
                min_xMin["width"] = width
            
            if rows > max_rows["val"]:
                max_rows["val"] = rows
                max_rows["char"] = char_c
                max_rows["top"] = yMax
            
        # for debug
        if debug:
            print("max_width:(%s=%d),max_yMax:(%s=%d), max_yMin:(%s=%d), min_xMin:(%s=%d)"%(
                max_width["char"],max_width["val"], max_yMax["char"], max_yMax["val"], 
                max_yMin["char"], max_yMin["val"], min_xMin["char"], min_xMin["val"]
            ))
        # 获取 origin的 row坐标
        yMax_val, yMin_val = max_yMax["val"], max_yMin["val"]

        if (yMax_val + yMin_val) >= font_size:
            row_pad = 0
        else:
            row_pad = (font_size - yMax_val - yMin_val) // 2
        origin_row = row_pad + yMax_val
        if origin_row - max_rows["top"] < 0:
            origin_row = max_rows["top"]

        # 获取 origin 的col 坐标
        widthMax = max_width["val"]
        assert widthMax <= font_size, "max_width(%d)-%s, must be smaller than fontsize(%d)"%(
            widthMax, max_width["char"], font_size
        )

        col_pad = (font_size -  widthMax) //2   # padding >= 0
        origin_col = col_pad - max_width["left"] 

        if origin_col + min_xMin["val"] < 0:
            origin_col = -min_xMin["val"]

        origin_point =  {"origin_row":origin_row, "origin_col":origin_col}

        if debug:
            print(origin_point)

        return origin_point
    
    
def test_v4():
    """ 遍历一遍，获取最大的width """

    ttf_file = "/home/yue/DataSets/Font_Data_v4/HandV4_1018/少女的祈祷.ttf"

    hand_charset = "./data/font_strokes_v2_lv1_release.txt"
    with open(hand_charset, "r") as f:
        charset = [line.strip("\n")[0] for line in f.readlines()][:100]
    
    g = Font_Generator_v4(charset, font_size=256)
    
    np_dict =  g.get_plain(ttf_file)
    for style_name, style_dict in np_dict.items():
        np_arrs =  style_dict["renderImg"]
        err_list = style_dict["err"]
        save_np_arrs(np_arrs, g.save_dir, phase="v4")
    return np_arrs

####################################################
# Copied from : /home/yue/DeepLearning/Diffusions/rewrite_diffusion_git/make_dataset
# time : 2024-10-09
# Make Data V3
####################################################
class Font_Generator_v3():
    def __init__(self,
                 human_label: Union[str, list] = None, 
                 font_size: int = 256,
                 save_path:str="./font_data/"):
        
        # load generate character into list
        if human_label is None:
            human_label = []
        elif isinstance(human_label, str):
            with open(human_label, 'r') as f:
                human_label = [line.strip("\n") for line in f.readlines()]
        
        self.human_label = human_label
        self.font_size = font_size
        self.save_path = save_path
        self.save_dir = ""
        
    def adaptive_fontSize(self, fontSize, ttf_face, char_list):
        """ 
            if image size 32, then board = 1
            if image size 64, then board = 4
            if image size 80 ,then board = 5;
            if image size 128,then board = 6;
            if image size 224,then board = 8;
            if image size 256,then board = 10;
            if image size 512,then board = 27;
        """

        if fontSize == 32:
            board = 1
        elif fontSize == 64:
            board = 4
        elif fontSize == 80:
            board = 5
        elif fontSize == 128 :
            board = 6
        elif fontSize == 224:
            board = 8
        elif fontSize == 256:
            board = 10
        elif fontSize == 512:
            board = 27

        fsize = fontSize
        face  = ttf_face

        def getFont_maxValue(face, char_list, fsize, fontSize):

            maxValue = 0

            for idxx, c in enumerate(char_list):

                face.set_char_size(fsize*64)
                flags = FT_LOAD_RENDER

                if face.get_char_index(ord(c)) == 0:        # filter NOGLPYH 
                    continue

                try: face.load_char(c, flags) 
                except:
                    print("Font- face load char [%s] failed !!!"%( c))
                    continue

                bitmap = face.glyph.bitmap

                maxValue_C = max(bitmap.width, bitmap.rows)
                if maxValue_C == 0:
                    print(" This char :train_char {} not in font".format(c))
                    continue

                if maxValue_C >=fontSize+int(fontSize/5):
                    char_list.remove(c) # if  character c so bigger then remove it!!
                    print("remove this char [%s] it so big size[%d]"%(c, maxValue_C))
                    continue

                if maxValue_C >= maxValue : 
                    maxValue = maxValue_C
                    maxChar = c
                    maxid = idxx

            print(" max Char is [C%d-%s] - size=%d "%(maxid, maxChar, maxValue))
            return maxValue

        
        maxValue_C = getFont_maxValue(face, char_list, fsize, fontSize)

        if maxValue_C > (fontSize-board):
            while maxValue_C >= (fontSize-board) :
                fsize -=1 
                maxValue_C = getFont_maxValue(face, char_list, fsize, fontSize)
        else:
            while maxValue_C < (fontSize - board*2):
                fsize +=1 
                maxValue_C = getFont_maxValue(face, char_list, fsize, fontSize)

        return fsize
    

    def render_to_numpy(self, fit_fontsize, origin_fontsize, font_face, char_list, bar_str, debug=False):
        
        c_sz = origin_fontsize

        face = font_face
        flags = FT_LOAD_RENDER
        err_list, np_arrs = [], []

        # write train dataset
        for  idxx, c in tqdm(enumerate( char_list)):
            if face.get_char_index(ord(c)) == 0:        # filter NOGLPYH 
                err_list.append(c)
                continue


            try: face.load_char(c, flags)
            except:
                print("Font- face load char [%s] failed !!!"%(c))
                err_list.append(c)
                continue
            bitmap = face.glyph.bitmap

            maxValue = bitmap.width if bitmap.width >= bitmap.rows else bitmap.rows
            if maxValue == 0:
                print(" This char :train_char {} not in font".format(c))
                err_list.append(c)
                continue
                
            width, rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
            top, left = face.glyph.bitmap_top, face.glyph.bitmap_left
            Z2 = np.zeros((rows, width))
            dx ,dy = 0, 0
            data = np.array(bitmap.buffer[:rows*pitch]).reshape(rows, pitch)
            data = 255 - data       # 黑底白字 转 成白底黑字
            Z2[dx:dx+rows, dy:dy+width] = data[:, :width]

            image = np.ones((c_sz, c_sz),dtype=np.uint8)*255
            dx = (c_sz - rows) / 2
            dy = (c_sz - width) /2
            dx, dy = int(dx), int(dy)
            image[dx:dx+rows, dy:dy+width] = Z2

            np_arrs.append([c, image])

        return np_arrs, err_list


    def get_plain(self, ttf):
        char_list = self.human_label
        char_size = [self.font_size]

        self.np_images = {}
        num_chars = len(char_list)
        ttf_name = ttf.split("/")[-1].split(".")[0]
        bar_str = "[%s] n=[%d]"%(ttf.split("/")[-1], num_chars)
        assert os.path.isfile(ttf), "[%s] not a file"%ttf
        save_path = os.path.join(self.save_path, ttf.split("/")[-1].split(".")[0])
        self.save_dir = save_path
        os.makedirs(save_path, exist_ok=True)

        face = Face(ttf)
        render_imgs = []
        for c_sz in char_size:
            
            font_size = c_sz
            #  绘制中文字符
            font_size = self.adaptive_fontSize(c_sz, face, char_list)
            if font_size !=0 :
                face.set_char_size(font_size *64)
                bar_str += "-SZ[%d]"%font_size
                np_arrs, err_list =  self.render_to_numpy(font_size, c_sz, face, char_list, bar_str=bar_str, debug=False)
                render_imgs.append(np_arrs)
            else:
                print(bar_str + "no chinese glyph")

        out = {"renderImg": render_imgs, "err":err_list}
        return {str(ttf_name):out}

def test_v3():
    """ 遍历一遍，获取最大的width """

    ttf_file = "/home/yue/DataSets/Font_Data_v4/HandV4_1018/少女的祈祷.ttf"

    hand_charset = "./data/font_strokes_v2_lv1_release.txt"
    with open(hand_charset, "r") as f:
        charset = [line.strip("\n")[0] for line in f.readlines()][:100]
    
    g = Font_Generator_v3(charset, font_size=256)
    
    np_dict =  g.get_plain(ttf_file)
    for style_name, style_dict in np_dict.items():
        np_arrs =  style_dict["renderImg"]
        err_list = style_dict["err"]

        save_np_arrs(np_arrs, g.save_dir, phase="v3")
    return np_arrs


def save_np_arrs(np_arrs, save_path, phase="v3"):
    if len(np_arrs) == 1:
        np_arrs = np_arrs[0]
    
    for c, np_img in np_arrs:
        cv2.imwrite("%s/c%s_%s.png"%(save_path, c, phase), np_img)

    

if __name__ == "__main__":

    # test_v4()
    test_v3()
    pass
    # mission_configs = [get_config("CLIP_ZeroShot_Test")]

    # for config in mission_configs:
        
    #     label_file = config["label_file"]

    #     with open(label_file, "r") as lf:
    #         char_label = [line.strip("\n") for line in lf.readlines()]

    #     g = Font_Generator(config, char_label)

    #     mode = 2
    #     if mode == 1:
    #         for ttf in tqdm(g.font_files, total=len(g.font_files)):
    #             g.get_plain(ttf)

    #         print("Done")
    #     elif mode == 2:
    #         # method 2
    #         p = Pool()
    #         p.map(g.get_plain, g.font_files)
    #         p.close()
    #         p.join()

