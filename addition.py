import cv2
import numpy as np


def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d]) 


def is_similar(point1, point2, threshold=1.0):
    return np.linalg.norm(point1-point2) < threshold

def corner_points(contours, img, grey, points_img):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    biggest_corners = []
    second_corners = []

    for i in range(len(contours)):
        img = img.copy()
        grey = np.float32(grey)

        mask = np.zeros(grey.shape, dtype="uint8")
        cv2.fillPoly(mask, [contours[i]], (255,255,255))
        dst = cv2.cornerHarris(mask,5,3,0.04)
        _, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(grey,np.float32(centroids),(5,5),(-1,-1),criteria)

        unique_corners = []
        for j in range(1, len(corners)):
            if not any(is_similar(corners[j], uc) for uc in unique_corners):
                unique_corners.append(corners[j])

        for c in unique_corners:
            cv2.circle(points_img, (int(c[0]), int(c[1])), radius=10, color=(0, 0, 255), thickness=-1)

        if i == 0:
            biggest_corners.append(list(unique_corners))
        else:
            second_corners.append(list(unique_corners))
 
    biggest_corners = np.array(biggest_corners).reshape(-1, 2)
    second_corners = np.array(second_corners).reshape(-1, 2)

    return points_img, biggest_corners, second_corners


def cut(img, corners):
    corners = sorted(corners, key=lambda x: x[1]) 
    top = sorted(corners[:2], key=lambda x: x[0])
    bottom = sorted(corners[2:], key=lambda x: x[0], reverse=True)

    corners = np.float32([top[0], top[1], bottom[0], bottom[1]])

    # corners of the entire image
    new_corners = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])

    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    cut_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    return cut_img


def get_choices(img, ques, choi):
    rows = np.vsplit(img, ques)
    choices = []
    for row in rows:
        cols = np.hsplit(row, choi)
        for choice in cols:
            choices.append(choice)

    choices = np.array(choices)
    _, h, w, c = choices.shape
    return choices.reshape((ques, choi, h, w, c))


def show_answers(img, answers, RESULTS, ques, choi):
    width = int(img.shape[1]/ques)
    height = int(img.shape[0]/choi)

    for i in range(ques):
        ans = answers[i]
        res = RESULTS[i]

        x = (ans * width) + width // 2
        y = (i * height) + height // 2

        if ans == res:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            cv2.circle(img, ((res * width) + width // 2, y), 20, (0, 255, 0), cv2.FILLED)
        
        cv2.circle(img, (x, y), 50, color, cv2.FILLED)
    
    return img


def put_back(img, corners, target_img):
    img_corners = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])

    corners = sorted(corners, key=lambda x: x[1]) 
    top = sorted(corners[:2], key=lambda x: x[0])
    bottom = sorted(corners[2:], key=lambda x: x[0], reverse=True)
    corners = np.float32([top[0], top[1], bottom[0], bottom[1]])

    matrix = cv2.getPerspectiveTransform(img_corners, corners)
    warped_img = cv2.warpPerspective(img, matrix, (target_img.shape[1], target_img.shape[0]))

    mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

    target_img = cv2.bitwise_and(target_img, target_img, mask=mask)
    result = cv2.add(target_img, warped_img)

    return result


def cut_grade(img, corners):
    corners = sorted(corners, key=lambda x: x[1]) 
    top = sorted(corners[:2], key=lambda x: x[0])
    bottom = sorted(corners[2:], key=lambda x: x[0], reverse=True)

    corners = np.float32([top[0], top[1], bottom[0], bottom[1]])

    # Calculate the width and height of the new image
    width = np.sqrt(((bottom[0][0]-bottom[1][0])**2)+((bottom[0][1]-bottom[1][1])**2))
    height = np.sqrt(((bottom[0][0]-top[0][0])**2)+((bottom[0][1]-top[0][1])**2))

    # corners of the new image
    new_corners = np.float32([[0, 0], [width-1, 0], [width-1, height-2], [0, height-1]])

    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    cut_img = cv2.warpPerspective(img, matrix, (int(width), int(height)))

    return cut_img


