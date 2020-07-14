import cv2 as cv
import PIL
#from skimage.color import rgb2gray, gray2rgb
import os

def splitData(root):
    """
    input: the test data, with with_hole.jpgs & seg.jpgs & mask.jpgs
    which are directly unzipped from the competition site
    output: lists which seperate these three types of images

    total: 464 test image .
    """
    tags = ['with_holes', 'mask', 'seg']

    inputFiles = []
    maskFiles = []
    segFiles = []

    for root, dirs, files in os.walk(root, topdown=False):
        for name in files:
            if tags[0] in name:
                inputFiles.append(os.path.join(root, name))
            elif tags[1] in name:
                maskFiles.append(os.path.join(root, name))
            else:
                segFiles.append(os.path.join(root, name))

    print("{}\nTotal capacity:\t{}".format(maskFiles, len(maskFiles)))

    return inputFiles, maskFiles, segFiles

def Semantic2Canny(img):
    (B,G,R) = cv.split(img)
    edge_B = cv.Canny(B, 80, 160)
    edge_G = cv.Canny(G, 80, 160)
    edge_R = cv.Canny(R, 80, 160)
    semantic_edge = edge_R + edge_G + edge_B
    return semantic_edge

def mergeOne(semantic_img_path):
    """
    input: a seg file's path
    function: write a image with "edge_semantic merged image to the input's same folder, replace 'seg' in the name with 'merge_es'
    """
    merge_es_path = semantic_img_path.replace("seg", "merge_es")

    semantic = cv.imread(semantic_img_path)
    semantic_edge = Semantic2Canny(semantic)

    # ele1: gray scaled iamge
    semantic = cv.cvtColor(semantic, cv.COLOR_BGR2GRAY)
    # ele2: edge image, with 
    ret,binary =cv.threshold(semantic_edge,125,255,cv.THRESH_BINARY)
    #print(semantic.shape)
    #print(edge)

    merge = semantic 
    # replace the edge part with 255, enhance the edge information 
    h, w = semantic.shape[0], semantic.shape[1]
    for i in range(h):
        for j in range(w):
            if binary[i,j] == 255:
                merge[i,j] = 255
            else:
                continue
    cv.imwrite(merge_es_path, merge)

def generateMerge(seg_files):
    """
    input: the path list of segmentation images
    output: write with edge_semantic_merged images
    """
    for seg_file in seg_files:
        print("converting data:\t{}".format(seg_file))
        mergeOne(seg_file)

def main():
    i_list, m_list, s_list = splitData("./for_users")
    generateMerge(s_list)

if __name__ == '__main__':
    main()
