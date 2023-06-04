import nltk
from pyvi import ViPosTagger , ViTokenizer
import re
from nltk.tokenize import word_tokenize , sent_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from wordcloud import STOPWORDS
class BaseVNPreprocessor:
    def __init__(self):
        self.emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)
        with open(os.path.join("NewsAPI" ,"vietnamese-stopword_dashed.txt") , encoding='utf-8') as f:
                 self.stopwords = f.readlines()
        self.vowel_table = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                    ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                    ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                    ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                    ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                    ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                    ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                    ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                    ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                    ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                    ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                    ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
        self.vowel_to_ids = {}
        for i in range(len(self.vowel_table)):
            for j in range(len(self.vowel_table[i]) - 1):
                self.vowel_to_ids[self.vowel_table[i][j]] = (i, j)
        self.uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
        self.unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    def loaddicchar(self):
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic
    # Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
    def convert_unicode(self ,txt):
        dicchar = self.loaddicchar()
        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()], txt)
    #xử lý sign or old
    def normalize_sign_of_a_word(self , word):
        if not self.is_valid_vietnam_word(word):
            return word
        chars = list(word)
        dau_cau = 0
        vowel_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.vowel_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.vowel_table[x][0]
            if not qu_or_gi or index != 1:
                vowel_index.append(index)
        if len(vowel_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.vowel_to_ids.get(chars[1])
                    chars[1] = self.vowel_table[x][dau_cau]
                else:
                    x, y = self.vowel_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.vowel_table[x][dau_cau]
                    else:
                        chars[1] = self.vowel_table[5][dau_cau] if chars[1] == 'i' else self.vowel_table[9][dau_cau]
                return ''.join(chars)
            return word
    
        for index in vowel_index:
            x, y = self.vowel_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.vowel_table[x][dau_cau]
                # for index2 in vowel_index:
                #     if index2 != index:
                #         x, y = vowel_to_ids[chars[index]]
                #         chars[index2] = vowel_table[x][0]
                return ''.join(chars)
    
        if len(vowel_index) == 2:
            if vowel_index[-1] == len(chars) - 1:
                x, y = self.vowel_to_ids[chars[vowel_index[0]]]
                chars[vowel_index[0]] = self.vowel_table[x][dau_cau]
                # x, y = vowel_to_ids[chars[vowel_index[1]]]
                # chars[vowel_index[1]] = vowel_table[x][0]
            else:
                # x, y = vowel_to_ids[chars[vowel_index[0]]]
                # chars[vowel_index[0]] = vowel_table[x][0]
                x, y = self.vowel_to_ids[chars[vowel_index[1]]]
                chars[vowel_index[1]] = self.vowel_table[x][dau_cau]
        else:
            # x, y = vowel_to_ids[chars[vowel_index[0]]]
            # chars[vowel_index[0]] = vowel_table[x][0]
            x, y = self.vowel_to_ids[chars[vowel_index[1]]]
            chars[vowel_index[1]] = self.vowel_table[x][dau_cau]
            # x, y = vowel_to_ids[chars[vowel_index[2]]]
            # chars[vowel_index[2]] = vowel_table[x][0]
        return ''.join(chars)
    
    
    def is_valid_vietnam_word(self , word):
        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, y = self.vowel_to_ids.get(char, (-1, -1))
            if x != -1:
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return False
                    vowel_index = index
        return True
    
    def normalize_sign_of_a_sen(self , sentence):
        """
            Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
            :param sentence:
            :return:
            """
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(r'(^\{P}*)([{L}.]*\{L}+)(\{P}*$)', r'\1/\2/\3', word).split('/')
            if len(cw) == 3:
                cw[1] = self.normalize_sign_of_a_word(cw[1])
            words[index] = ''.join(cw)
        return ' '.join(words)
    def remove_special_char(self, document):
        special_characters=['@','#','$','*','&']
        normal_string=document
        for i in special_characters:
    # Replace the special character with an empty string
            normal_string=normal_string.replace(i,"")
        return normal_string
    def remove_emoji(self, document):
        return self.emoji_pattern.sub(r'', document)
    def remove_url(self, document):
        url_pattern = re.compile(r'http\S+')
        return url_pattern.sub(r'', document)

    def remove_stopwords(self , doc):
        self.stopwords = [i.replace("\n","") for i in self.stopwords]
        return [w.lower() for w in doc if w not in self.stopwords]
    
def text_preprocess(document):
    """
        Tiền xử lý document
        --------
        input: nhận vào một document là một chuỗi ký tự
        output: một mảng các tokens có được từ tiền xử lý document

    """
    preprocessor = BaseVNPreprocessor()
    # # normalize unicode
    document = preprocessor.convert_unicode(document)
    
    # # normalize sign 
    document = preprocessor.normalize_sign_of_a_sen(document)

    # remove special characters
    document = preprocessor.remove_special_char(document)

    # remove emoji characters and url
    document = preprocessor.remove_emoji(document)
    document = preprocessor.remove_url(document)
    # split words
    document = ViTokenizer.tokenize(document)

    # remove unnecessary words
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)

    # # remove extra space
    document = re.sub(r'\s+', ' ', document).strip()
    
    # tokenize 
    document = word_tokenize(document)
    
    # remove stopwords
    document = preprocessor.remove_stopwords(document)
    # giả sử bộ dữ liệu có từ sai chính tả cần sửa lỗi chính tả
    # bộ ngữ liệu chỉ tập trung vào từ
    return document