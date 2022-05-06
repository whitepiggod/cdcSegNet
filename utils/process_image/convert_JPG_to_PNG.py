from PIL import Image
import os

paths = os.listdir('./images/train/image')
print(paths)
for path in paths:
    if path[-3:] == 'jpg':
        print('./images/train/image/' + path[0:-3] + 'jpg')
        os.remove('./images/train/image/' + path[0:-3] + 'jpg')
    # im = Image.open('./images/train/image/' + path)
    # print('./images/train/image/' + path[0:-3] + 'png')
    # im.save('./images/train/image/' + path[0:-3] + 'png')
    # os.remove('./images/train/image/' + path[0:-3] + 'jpg')
# im = Image.open('Foto.jpg')
# im.save('Foto.png')