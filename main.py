import cv2
import numpy as np
from addition import concat_vh, corner_points, cut, get_choices, show_answers, put_back, cut_grade

###############################
PATH = "img/1.png"
DIMENSIONS = (700, 700)
RESULTS = [1,2,0,3,3]
QUESTION_CHOICE_COUNT = (5, 5)
###############################


def process_image(path, dimensions):
    try:
        img = cv2.imread(path)
    except:
        print("Cannot read file")
        return None

    img = cv2.resize(img, dimensions)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 1)
    canny = cv2.Canny(blur, 10, 50)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 8)

    return img, contour_img, contours, grey, blur, canny



def calculate_points(choices, results):
    white_pixels_matrix = np.zeros(QUESTION_CHOICE_COUNT)
    for i, row in enumerate(choices):
        for j, image in enumerate(row):
            white_pixels = np.sum(image == [255, 255, 255])
            white_pixels_matrix[i][j] = white_pixels

    answers = []
    for row in white_pixels_matrix:
        chosen = np.argmax(row)
        answers.append(chosen)
    print(answers, results)

    points = 0
    for i, choice in enumerate(answers):
        if choice == results[i]:
            points += 1
    points *= int(100/len(answers))
    print(points)
    return points, answers



def main():
    img, contour_img, contours, grey, blur, canny = process_image(PATH, DIMENSIONS)
    points_img, biggest_corners, grade_corners = corner_points(contours, contour_img, grey, img.copy())

    cut_img = cut(img.copy(), biggest_corners)

    grey_cut = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
    thr_cut = cv2.threshold(grey_cut, 180, 255, cv2.THRESH_BINARY_INV)[1]
    thr_cut = cv2.cvtColor(thr_cut, cv2.COLOR_GRAY2BGR)

    choices = get_choices(thr_cut.copy(), *QUESTION_CHOICE_COUNT)
    points, answers = calculate_points(choices, RESULTS)
    ans_img = show_answers(cut_img.copy(), answers, RESULTS, *QUESTION_CHOICE_COUNT)

    black_ans_img = show_answers(np.zeros_like(ans_img), answers, RESULTS, *QUESTION_CHOICE_COUNT)

    ori_img = put_back(black_ans_img, biggest_corners, np.zeros_like(img))

    final_img = put_back(black_ans_img, biggest_corners, img.copy())
    grade_img = cut_grade(img.copy(), grade_corners)
    cv2.putText(grade_img, str(points)+"%", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
    final_img = put_back(grade_img, grade_corners, final_img)

    # blank = np.zeros_like(img)
    grey3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    blur3 = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    canny3 = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    array = [[img, grey3, blur3, canny3],
             [contour_img, points_img, cut_img, thr_cut],
             [ans_img, black_ans_img, ori_img, final_img]]
    show = concat_vh(array)

    cv2.imshow("original", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
