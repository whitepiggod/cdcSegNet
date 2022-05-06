import os

paths = os.listdir('./images/train/mask')
paths1 = os.listdir('./images/train/image')

# 删除不对应的文件
for path in paths1:
    # newPath = path[0:-3] + 'jpg'
    if (path in paths):
        print('true')
    else:
        # print('false')
        print(path)
        os.remove('./images/train/image/' + path)

# 修改文件名
# fileList = os.listdir('./images/train/mask')
# dir = './images/train/mask/'
# n = 0
# for path in fileList:
#     # 设置旧文件名（就是路径+文件名）
#     # oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
#
#     # 设置新文件名
#     newname = '{}.png'.format(n)
#     os.rename(dir + path, dir + newname)  # 用os模块中的rename方法对文件改名
#     # print(oldname, '======>', newname)
#
#     n += 1