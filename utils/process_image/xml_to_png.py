import os
import SimpleITK
import pydicom
import numpy as np
import glob
from tqdm import tqdm
import bs4
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
import cv2
import xml


dataset_size = 300000  # 生成的数据集大小



def is_dicom_file(filename):
    '''
       判断某文件是否是dicom格式的文件
    :param filename: dicom文件的路径
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def load_patient(dir_list,type):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径,，！最终文件夹
    :return: dicom list
    '''

    uids=[]
    # print(dir_list)
    if (type == "nodule"):
        uids,nodule_pos = find_nodule_frame(dir_list)
    else:
        uids, nodule_pos = find_non_nodule_frame(dir_list)
    if (uids == None):
        return None, None
    slices = [0] * len(uids)

    file_list = os.listdir(dir_list)
    for file in file_list:
        if file.endswith("xml"):
            continue
        file_path = os.path.join(dir_list, file)
        ds = pydicom.dcmread(file_path)
        #print(ds.SOPInstanceUID)
        if (ds.SOPInstanceUID in uids):
            #print(file)# 这里的0x0008,0x0018是SOP_Instance_UID
            #slices.append(pydicom.read_file(file_path))
            slices[uids.index(ds.SOPInstanceUID)]=pydicom.read_file(file_path)
            #slices.insert(uids.index(ds.SOPInstanceUID),pydicom.read_file(file_path))
            #slices.remove(0)

            # print(count)

    ol=slices.count(0)
    lop=len(slices)
    if (slices.count(0) == len(slices)):
        return None,None


    #莫名其妙！有些CT在xml文件中有但是在dcm里面找不到，可能是我没下完全叭，梯子太抖了
    #所以这里做一个筛选
    real_slices=[]
    real_nodule_pos=[]
    for i in range(len(uids) - 1):
        if(slices[i] is 0):
            continue
        else:
            real_slices.append(slices[i])
            real_nodule_pos.append(nodule_pos[i])

    for i in range(len(real_slices)-1):
        if(uids[i]!=real_slices[i].SOPInstanceUID):
            print("not correct")
            print(i)
            #os._exit(0)
            return None, None
    return real_slices, real_nodule_pos



def get_pixels_hu_by_simpleitk(slices):
    '''
        读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    # reader = SimpleITK.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    # reader.SetFileNames(dicom_names)
    # image = reader.Execute()
    # img_array = SimpleITK.GetArrayFromImage(image)
    # img_array[img_array == -2000] = 0
    # return img_array

    img_array = []
    for sli in slices:
        img_array.append(np.array(sli.pixel_array))
    img_array = np.array(img_array)
    img_array[img_array == -2000] = 0
    img_array = img_array - 1024           #################################全场最佳！ -1024 不减图片就全是黑的！

    return np.array(img_array)



def normalize_hu(image):
    '''
    将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    :param image 输入的图像数组
    :return: 归一化处理后的图像数组
    '''
    MIN_BOUND = -1000
    MAX_BOUND = 400
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def load_patient_images(src_dir, wildcard="*.*", exclude_wildcards=[]):
    '''
    读取一个病例的所有png图像，返回值为一个三维图像数组
    :param image 输入的一系列png图像
    :return: 三维图像数组
    '''
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1,) + im.shape) for im in images]
    res = np.vstack(images)
    return res

def get_segmented_lungs(im):
    '''
    对输入的图像进行肺部区域分割，提取有效的肺部区域，用于模型训练
    :param 输入的图像
    :return: 返回分割结果
    '''

    binary = im < -400  # Step 1: 转换为二值化图像
    cleared = clear_border(binary)  # Step 2: 清除图像边界的小块区域
    label_image = label(cleared)  # Step 3: 分割图像

    areas = [r.area for r in regionprops(label_image)]  # Step 4: 保留2个最大的连通区域
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    selem = disk(2)  # Step 5: 图像腐蚀操作,将结节与血管剥离
    binary = binary_erosion(binary, selem)
    selem = disk(10)  # Step 6: 图像闭环操作,保留贴近肺壁的结节
    binary = binary_closing(binary, selem)
    edges = roberts(binary)  # Step 7: 进一步将肺区残余小孔区域填充
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0  # Step 8: 将二值化图像叠加到输入图像上
    im[get_high_vals] = -2000
    print('lung segmentation complete.')
    return im, binary



# 找出所有有结节的frame的UID

def find_nodule_frame(folder_path):
    '''
    folder_path:最终存放dcm文件的文件夹路径
            --folder_path
                -xx.dcm
                -xxx.dcm
                ...
    '''
    file_list = os.listdir(folder_path)
    xml_path=folder_path
    for file in file_list:
        if file.endswith("xml"):
            file_path = os.path.join(folder_path, file)
            xml_path = file_path
    if(xml_path==folder_path):
        return None,None

    uids = []
    nodule_pos = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = bs4.BeautifulSoup(markup, features="xml")
    reading_session = xml.LidcReadMessage.find_all("readingSession")

    for readingSession in reading_session:
        # unblindedReadNodule为医生标注的结节信息nonNodule
        nodule_info = readingSession.find_all("unblindedReadNodule")
        for nodule_info in nodule_info:
            nodule_roi = nodule_info.find_all("roi")
            for nodule_roi in nodule_roi:
                edge_map=nodule_roi.find_all("edgeMap")
                one_nodule=[]
                frame_uid = nodule_roi.find_all("imageSOP_UID")
                if(frame_uid[0].string in uids):                      # 因为有多个医生的诊断信息  所以会出现重复UID
                    continue
                else:
                    uids.append(frame_uid[0].string)
                    for edge in edge_map:
                        one_nodule.append([edge.find("xCoord").text, edge.find("yCoord").text])


                    nodule_pos.append(one_nodule)



    #nodule_pos里面装的是edgeMap，与uid是一一对应的
    return uids,nodule_pos


def find_non_nodule_frame(folder_path):
    '''
    folder_path:最终存放dcm文件的文件夹路径
            --folder_path
                -xx.dcm
                -xxx.dcm
                ...
    '''
    file_list = os.listdir(folder_path)
    xml_path=folder_path
    for file in file_list:
        if file.endswith("xml"):
            file_path = os.path.join(folder_path, file)
            xml_path = file_path
    if(xml_path==folder_path):
        return None,None

    uids = []
    nodule_pos = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = bs4.BeautifulSoup(markup, features="xml")
    reading_session = xml.LidcReadMessage.find_all("readingSession")

    for readingSession in reading_session:
        # unblindedReadNodule为医生标注的结节信息nonNodule
        nodule_info = readingSession.find_all("nonNodule")
        for nodule_info in nodule_info:
                edge_map=nodule_info.find_all("locus")
                one_nodule=[]
                frame_uid = nodule_info.find_all("imageSOP_UID")
                if(frame_uid[0].string in uids):                      # 因为有多个医生的诊断信息  所以会出现重复UID
                    continue
                else:
                    uids.append(frame_uid[0].string)
                    for edge in edge_map:
                        one_nodule.append([edge.find("xCoord").text, edge.find("yCoord").text])


                    nodule_pos.append(one_nodule)



    #nodule_pos里面装的是edgeMap，与uid是一一对应的
    return uids,nodule_pos


if __name__ == '__main__':
    dicom_dir=r"./datasets/111/manifest-1600709154662/LIDC-IDRI"
    # dicom_dir= r"./dicom_demo"
    # count = 7631
    count = 1
    dir_list = []
    for root, dirs, files in os.walk(dicom_dir):
        # 10是根据数据集特点写的，因为一个病患有两种ct文件，一个是胸部横截面，一个是正面CT，正面的只有两张
        if (len(files) > 10 and dataset_size  > 0):
            dataset_size -= len(files)
            dir_list.append(root)

    for file_dir in dir_list:
        print(file_dir)
        # slices, edgeMap = load_patient(file_dir,"nodule")
        # 提取非结节的部分
        slices, edgeMap = load_patient(file_dir, "non-nodule")
        # slices, edgeMap = non_nodule_slices + nodule_slices, non_nodule_edgeMap + nodule_edgeMap
        # print(len(slices))
        # print(len(edgeMap))
        if(slices==None ):
            continue
        # 提取dicom文件中的像素值
        image = get_pixels_hu_by_simpleitk(slices)
        # print("")
        for i in tqdm(range(image.shape[0])):
            # if(len(edgeMap[i])<10):  #控制结点大小
            #     continue
            # lung_mask_path="./dataset3.0/lung_masks/lungMask_"+ str(count) + ".png"
            # lung_parenchyma_path="./dataset3.0/lung_parenchyma/lungparen_"+ str(count) + ".png"
            # lung_image_path="./dataset3.0/lung_images/lungImg_"+ str(count) + ".png"
            # lung_parenchyma_white_path = "./dataset3.0/lung_parenchyma_white/lungparen_" + str(count) + ".png"
            lung_image_path = "./dataset3.0/image/non_nodule" + str(count) + ".png"
            org_img = normalize_hu(image[i])
            # 保存图像数组为灰度图(.png)
            cv2.imwrite(lung_image_path, org_img * 255)

            im, binary = get_segmented_lungs(image[i])  # 肺部区域分割   这一句不隐藏 保存的就是肺实质图片
            # 将像素值归一化到[0,1]区间
            org_img = normalize_hu(image[i])
            # 保存图像数组为灰度图(.png)
            # cv2.imwrite(lung_parenchyma_path, org_img * 255)
            # cv2.imwrite(lung_parenchyma_white_path, im * 255)
            # cv2.imwrite(lung_mask_path, binary * 255)


            # nodule_mask_path = "./dataset3.0/nodule_masks/noduleMask_" + str(count) + ".png"
            nodule_mask_path = "./dataset3.0/mask/non_nodule" + str(count) + ".png"
            img = np.zeros([512, 512], np.uint8)
            edge = np.int32([edgeMap[i]])
            cv2.polylines(img, edge, 1, 1)
            cv2.fillPoly(img, edge, 1)
            cv2.imwrite(nodule_mask_path, img * 255)

            count += 1

