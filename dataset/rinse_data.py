"""文本去除所有特殊字符"""
import re
import os

pro_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# text_path = os.path.join(pro_path, "data")
# # 拿到文件夹下面的所有文件
# text_list = os.listdir(text_path)
# for path in text_list:
#     with open(os.path.join(pro_path, path), 'r') as f:
#         result = f.read()
#         # 使用正则表达式去匹配标点符号
#         result = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", result)
#         print(result)
#         # result1 = demo(result)
#         with open(os.path.join(pro_path, path), 'w') as w:
#             w.write(str(result))

text_path = os.path.join(pro_path, "data/wiki.txt")

with open(os.path.join(pro_path, text_path), 'r', encoding='utf-8') as f:
    result = f.read()
    # 使用正则表达式去匹配标点符号
    result = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", result)
    print(result)
    # result1 = demo(result)
    with open(os.path.join(pro_path, "data/r_wiki.txt"), 'w', encoding='utf-8') as w:
        w.write(str(result))
