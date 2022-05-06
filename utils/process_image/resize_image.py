import os
import os.path
from PIL import Image
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout, width, height, type):
  img = Image.open(filein)
  out = img.resize((width, height),Image.ANTIALIAS)
  #resize image with high-quality
  out.save(fileout, type)
if __name__ == "__main__":
  filein = './images/train/222/mask'
  fileout = './images/train/222/mask01'
  width = 512
  height = 512
  type = 'png'
  paths = os.listdir(filein)
  for path in paths:
    in_path = filein + '/' + path
    out_path = fileout + '/' + path
    ResizeImage(in_path, out_path, width, height, type)
  # ResizeImage(filein, fileout, width, height, type)