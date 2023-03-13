import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def contour_to_percent_points(contour: list, img_width: float, img_height: float):
    percent_points = [[point_wrapper[0][0]/img_height, point_wrapper[0][1]/img_width] for point_wrapper in contour]
   
    return percent_points

def points_format(points: list, extra: str):
    output = [f"{extra}"] + list(map(lambda point: f"{point[0]:.8f} {point[1]:.8f}", points))
    return " ".join(output)
    

def save_countours(path: str, contours: list, img_width: float, img_height: float, extra: str):
    output = []
    for contour in contours:
        output.append(points_format(contour_to_percent_points(contour, img_width, img_height), extra))
    with open(path, "w") as f:
        f.write("\n".join(output))
    


directory_name = "sample"

for filename in os.listdir(r"./"+directory_name):
    array_of_img = cv2.imread(directory_name + "/" + filename) # 3-channel image
    W = array_of_img.shape[0]  #480
    H = array_of_img.shape[1]  #640
    array_of_img1 = cv2.cvtColor(array_of_img, cv2.COLOR_BGR2GRAY)
    ret, bin_image = cv2.threshold(array_of_img1, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((11, 11), dtype=np.uint8)
    dilate = cv2.dilate(bin_image, kernel, 1)
    erosion = cv2.erode(dilate, kernel, iterations=1)

    mask = np.zeros([W+2, H+2],np.uint8)
    im_floodfill = erosion.copy()
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break

    cv2.floodFill(im_floodfill, mask,seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out1 = erosion | im_floodfill_inv

    kernel = np.ones((5, 5), dtype=np.uint8)
    erosion = cv2.erode(im_out1, kernel, iterations=1)
    im_out1 = cv2.dilate(erosion, kernel, 1)
    

    contours, hierarchy = cv2.findContours(im_out1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    # for contour in contours:
    #     percent_points = contour_to_percent_points(contour, W, H)
    #     plt.gca().invert_yaxis()
    #     plt.scatter(x=[point[0] for point in percent_points], y=[point[1] for point in percent_points])
    # plt.scatter(x = 0, y = 0)
    # plt.scatter(x = 0, y = 1)
    # plt.scatter(x = 1, y = 1)
    # plt.scatter(x = 1, y = 0)
    # plt.show()
    # plt.clf()
    # save_countours(filename[:-4] + '.txt', contours, W, H, "0")

    cv2.drawContours(array_of_img,contours,-1,(0,0,255),3)  
    cv2.imshow("img", array_of_img)  
    cv2.waitKey(0)  

    # print (len(contours))
    # plt.imshow(contours[0])
    # plt.imshow(array_of_img)
    # plt.show()
    # plt.imshow(im_out1)
    # plt.show()