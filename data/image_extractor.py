import cv2
import numpy as np
import os


import time


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None

# mouse callback function
point_list = []

s1 = (None, None)
s2 = (None, None)

scale = 1

road_width = 0.2 * 256


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)


def getEquidistantPoints(p1, p2, n):
    return [(lerp(p1[0], p2[0], 1. / n * i), lerp(p1[1], p2[1], 1. / n * i)) for i in range(n + 1)]


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing, s1, s2, scale
    if event == cv2.EVENT_MBUTTONUP:
        if s1 == (None, None):
            s1 = (x, y)
        elif s2 == (None, None):
            s2 = (x, y)
            cv2.line(img, s1, s2, color=(255, 0, 0,), thickness=3)
            scale = road_width / np.sqrt((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        # drawing = True
        if pt1_x is None:
            pt1_x, pt1_y = x, y
        else:
            p1, p2 = np.array([pt1_x, pt1_y]), np.array([x, y])
            distance = np.linalg.norm(p1 - p2)
            # print(distance)
            interpolated = getEquidistantPoints(p1, p2, int(distance / 10))
            for point in interpolated:
                print(point)
                point_list.append((point[0], point[1]))
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=3)
            pt1_x, pt1_y = x, y
    """elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=3)
            pt1_x, pt1_y = x, y
            point_list.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=3)
        point_list.append((x, y))"""


#name="ryan-searle-1jckkOd-GU8-unsplash.jpg"
# name="marcel-strauss-cmlaCF_F7Rw-unsplash.jpg"

# name = "path-road-route-aerial-view-drone-view-geology-164491-pxhere.com.jpg"
name = "style0.jpg"
seg_name = "style0_seg.jpg"
# name ="grass-plant-road-leaf-flower-botany-152310-pxhere.com.jpg"

path = "real_images/" + name
img = cv2.imread(path)
path2 = "real_images/" + seg_name
seg_img = cv2.imread(path2)
try:
    os.mkdir(f"patches/{name}")
except OSError:
    print("Already extracted delete directory to replace")
    quit()
else:
    print("Successfully created the directory %s " % path)


try:
    os.mkdir(f"patches/{seg_name}")
except OSError:
    print("Already extracted delete directory to replace")
    quit()
else:
    print("Successfully created the directory %s " % path2)
cv2.namedWindow('test draw', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('test draw', line_drawing)
#cv2.resizeWindow('test draw', 1024, 1024)
while (1):
    cv2.imshow('test draw', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

img = cv2.imread(path)

inv_scale = (1 / scale)


def AngleBtw2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return np.degrees(np.arctan2(changeInY, changeInX))



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


rotate = False
flip = False
follow_camera = True
c=0

for i, (x, y) in enumerate(point_list):
    if i == 0 or i>= len(point_list)-1:
        continue

    # (old_x, old_y) = point_list[i-1]
    # angle = AngleBtw2Points(np.array([x,y]), np.array([old_x, old_y]))
    s = img.shape
    # img = rotate_image(img, -angle)
    if follow_camera:
        x1 = int(x - inv_scale * 182)
        x2 = int(x + inv_scale * 182)
        y1 = int(y - inv_scale * 182)
        y2 = int(y + inv_scale * 182)
        if (x1 < 0 or y1 < 0 or y2 > s[0] or x2 > s[1]):
            print("too small")
            continue
        sub_img = img[y1:y2, x1:x2]
        sub_img_seg = seg_img[y1:y2, x1:x2]

        (old_x, old_y) = point_list[i - 1]
        (new_x, new_y) = point_list[i + 1]
        angle = AngleBtw2Points(np.array([new_x, new_y]), np.array([old_x, old_y]))
        sub_img = rotate_image(sub_img, angle -90)
        sub_img_seg = rotate_image(sub_img_seg, angle-90)



        x, y = sub_img.shape[0] / 2, sub_img.shape[1] / 2
        x1 = int(x - inv_scale * 127)
        x2 = int(x + inv_scale * 128)
        y1 = int(y - inv_scale * 127)
        y2 = int(y + inv_scale * 128)
        sub_img = sub_img[y1:y2, x1:x2]
        sub_img_seg = sub_img_seg[y1:y2, x1:x2]

    elif not rotate:
        x1 = int(x - inv_scale * 127)
        x2 = int(x + inv_scale * 128)
        y1 = int(y - inv_scale * 127)
        y2 = int(y + inv_scale * 128)
        if (x1 < 0 or y1 < 0 or y2 > s[0] or x2 > s[1]):
            print("too small")
            continue
        sub_img = img[y1:y2, x1:x2]
        sub_img_seg = seg_img[y1:y2, x1:x2]
    else:
        x1 = int(x - inv_scale * 182)
        x2 = int(x + inv_scale * 182)
        y1 = int(y - inv_scale * 182)
        y2 = int(y + inv_scale * 182)
        if (x1 < 0 or y1 < 0 or y2 > s[0] or x2 > s[1]):
            print("too small")
            continue
        sub_img = img[y1:y2, x1:x2]
        sub_img_seg = seg_img[y1:y2, x1:x2]
        angle = np.random.randint(-90, 90)
        sub_img = rotate_image(sub_img, angle)
        sub_img_seg = rotate_image(sub_img_seg, angle)

        x,y = sub_img.shape[0] / 2, sub_img.shape[1] / 2
        x1 = int(x - inv_scale * 127)
        x2 = int(x + inv_scale * 128)
        y1 = int(y - inv_scale * 127)
        y2 = int(y + inv_scale * 128)
        sub_img = sub_img[y1:y2, x1:x2]
        sub_img_seg = sub_img_seg[y1:y2, x1:x2]

    sub_img = cv2.resize(sub_img, dsize=(256, 256))
    sub_img_seg = cv2.resize(sub_img_seg, dsize=(256, 256))

    cv2.imshow("test", sub_img)

    cv2.imwrite(f"patches/{name}/patch_{c}.jpg", sub_img)
    cv2.imwrite(f"patches/{seg_name}/patch_{c}.jpg", sub_img_seg)
    c+=1
    if flip:
        fliped_hor = cv2.flip(sub_img, 1)
        cv2.imwrite(f"patches/{name}/patch_{c+1}.jpg", fliped_hor)
        fliped_ver = cv2.flip(sub_img, 0)
        cv2.imwrite(f"patches/{name}/patch_{c + 2}.jpg", fliped_ver)
        fliped_both = cv2.flip(sub_img, -1)
        cv2.imwrite(f"patches/{name}/patch_{c + 3}.jpg", fliped_both)

        fliped_hor = cv2.flip(sub_img_seg, 1)
        cv2.imwrite(f"patches/{seg_name}/patch_{c+1}.jpg", fliped_hor)
        fliped_ver = cv2.flip(sub_img_seg, 0)
        cv2.imwrite(f"patches/{seg_name}/patch_{c + 2}.jpg", fliped_ver)
        fliped_both = cv2.flip(sub_img_seg, -1)
        cv2.imwrite(f"patches/{seg_name}/patch_{c + 3}.jpg", fliped_both)

        cv2.waitKey(100)
        c +=3

cv2.destroyAllWindows()
