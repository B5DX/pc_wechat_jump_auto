# 可能需要调整的参数：
# 1. window_left_top和window_right_bottom的值，保证截图区域准确。
# 2. temp_player.jpg和temp_end.jpg最好自己重新截图以匹配分辨率。
# 3. jump函数中的参数2.735以及moveTo鼠标移动位置。
# 4. 当距离过近时容易跳远，程序末尾进行了距离缩放。该参数可能需要调整。
import os
import cv2
import numpy as np
import time
import random
from PIL import ImageGrab
from matplotlib import pyplot as plt
import pyautogui
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

DEBUG = True
window_left_top = (734, 80)
window_right_bottom = (1185, 938)

def get_screenshot(id):
    screenshot = ImageGrab.grab(bbox=(*window_left_top, *window_right_bottom))
    screenshot.save(f'history/{id}_rgb.png')
    return screenshot

def jump(distance):
    # 按压时间，这个参数还需要针对屏幕分辨率进行优化
    press_time = distance * 2.74
    press_time /= 1000

    # 移动鼠标模拟按压
    rand = random.randint(0, 9) * 4
    pyautogui.moveTo(window_left_top[0] * 0.2 + window_right_bottom[0] * 0.8 + rand, 
                     window_left_top[1] * 0.2 + window_right_bottom[1] * 0.8 + rand)
    pyautogui.mouseDown()
    time.sleep(press_time)
    pyautogui.mouseUp()


def get_center(img_canny, center1_loc):
    H, W = img_canny.shape
    # 利用边缘检测的结果寻找物块的上沿和下沿
    # 进而计算物块的中心点
    y_top = np.nonzero([max(row) for row in img_canny[int(H / 4): int(H * 2 / 3), int(W / 10): int (W * 9 / 10)]])[0][0] + int(H / 4)
    x_top = int(np.mean(np.nonzero(img_canny[y_top, int(W / 10): int (W * 9 / 10)]))) + int(W / 10)

    if x_top > center1_loc[0]: # board is on the right
        x_boarder = W * 14 // 15
        for x in range(x_boarder, x_top, -1):
            if np.sum(img_canny[y_top: center1_loc[1], x]) != 0:
                x_boarder = x
                break
        x_delta = x_boarder - x_top
    else:
        x_boarder = int(W / 20)
        success = False
        tried_times = 0
        while not success: # 多次搜索，逐渐减弱阴影判据，防止实体线条太少被当做阴影过滤掉
            for x in range(x_boarder, x_top):
                tmp = np.sum(img_canny[y_top: center1_loc[1], x])
                # print(x, tmp)
                if tmp != 0: 
                    # 左侧消除阴影影响
                    if tmp <= 255 * (12 - tried_times * 4) and np.sum(img_canny[y_top: center1_loc[1], x + 4]) <= (12 - tried_times * 4) * 255: 
                        continue
                    x_boarder = x
                    success = True
                    break
            tried_times += 1
            # print(f'Tried times: {tried_times}')
        x_delta = x_top - x_boarder
    if DEBUG:
        pass
        # print(f'x_top={x_top}, x_boarder={x_boarder}, delta={x_delta}')

    # 根据顶点搜索底部点。预先跳过一些y_bottom减轻白点和纹路的影响。
    y_bottom = y_top + max(20, x_delta)
    if x_delta > 120: # 很可能是异常情况，应该没有这么大的board
        y_bottom = y_top + 20 # 保守使用20作为跳过长度
    
    rows = list(range(y_bottom, H))
    idx = 0
    tmp = []
    while idx < len(rows):
        row = rows[idx]
        tmp.append(row)
        idx += 1
        if np.sum(canny_img[row, x_top - 2: x_top + 2]) != 0:
            y_bottom = row
            break

    x_center, y_center = x_top, (y_top + y_bottom) // 2
    if DEBUG:
        cv2.rectangle(img_canny, (x_boarder, y_top - 2), (x_boarder + 2, center1_loc[1]), 255, 1)
        cv2.rectangle(img_canny, (x_top, y_top), (x_top, y_bottom), 255, 2)
        cv2.rectangle(img_canny, (x_top - 2, y_top - 2), (x_top + 2, y_top + 2), 255, 2)
        for row in tmp:
            cv2.rectangle(img_canny, (x_top - 2, row - 2), (x_top + 2, row + 2), 255, 2)
        cv2.rectangle(img_canny, (x_top - 2, y_bottom - 2), (x_top + 2, y_bottom + 2), 255, 2)
        cv2.rectangle(img_canny, (x_center - 10, y_center - 10), (x_center + 10, y_center + 10), 255, 2)
        plt.imshow(img_canny)
        plt.savefig('history/%s_canny.png' % i)
        plt.clf()
    return img_canny, x_center, y_center, x_delta

# 匹配小跳棋的模板
temp1 = cv2.imread('temp_player.jpg', 0)
w1, h1 = temp1.shape[::-1]
# 匹配游戏结束画面的模板
temp_end = cv2.imread('temp_end.jpg', 0)

print('Game start!')
# 循环直到游戏失败结束
for i in range(5000):
    if i % 10 == 0:
        print("Round %d" % i)
    im = get_screenshot(i)

    img_rgb = cv2.imread(f'history/{i}_rgb.png', 0)

    # 如果在游戏截图中匹配到带"再玩一局"字样的模板，则循环中止
    res_end = cv2.matchTemplate(img_rgb, temp_end, cv2.TM_CCOEFF_NORMED)
    if cv2.minMaxLoc(res_end)[1] > 0.95:
        print('Game over!')
        break

    # 模板匹配截图中小跳棋的位置
    res1 = cv2.matchTemplate(img_rgb, temp1, cv2.TM_CCORR_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    center1_loc = (max_loc1[0] + w1 // 2, max_loc1[1] + int(h1 * 0.9))

    # 边缘检测
    # 以RGB格式重新读入图片，避免灰度图无法精确检测边缘
    img_rgb = cv2.imread(f'history/{i}_rgb.png', 1)
    img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    canny_img = cv2.Canny(img_rgb, 4, 10)
    H, W = canny_img.shape

    # 消去小跳棋轮廓对边缘检测结果的干扰
    for k in range(max_loc1[1] - h1 // 10, max_loc1[1] + int(h1 * 1.1)):
        for b in range(max_loc1[0] - w1 // 9, max_loc1[0] + int(w1 * 10 / 9)):
            canny_img[k][b] = 0
    img_rgb, x_center, y_center, x_delta = get_center(canny_img, center1_loc)

    distance = (center1_loc[0] - x_center) ** 2 + (center1_loc[1] - y_center) ** 2
    distance = distance ** 0.5
    if DEBUG:
        os.rename(f'history/{i}_canny.png', f'history/{i}_canny_{x_delta:.0f}_{distance:.0f}.png')
    # 距离过近时容易跳远，调整距离
    if distance < 140:
        # print(f'{distance}', end='->')
        if distance >= 100:
            distance = 0.9 * distance
        elif distance >= 85:
            distance = 0.85 * distance
        elif distance >= 70:
            distance = 0.75 * distance
        else:
            distance = 0.7 * distance
        # print(f'{distance}')
    else:
        pass
        # print(distance)
    
    jump(distance)
    
    # 等待1.5s避免受到动画影响
    time.sleep(1.5)

    if i >= 10:
        for f in os.listdir('history'):
            if f.split('_')[0] == str(i - 10):
                os.remove(f'history/{f}')
