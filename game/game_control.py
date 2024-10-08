import random
import time
from typing import Tuple

from adb.scrcpy_adb import ScrcpyADB
import math


class GameControl:
    def __init__(self, adb: ScrcpyADB):
        self.adb = adb

    def calc_mov_point(self, angle: float) -> Tuple[int, int]:
        angle = angle % 360
        rx, ry = (460, 850)
        r = 145

        x = rx + r * math.cos(angle * math.pi / 180)
        y = ry - r * math.sin(angle * math.pi / 180)
        return int(x), int(y)

    def move(self, angle: float, t: float):
        # 计算轮盘x, y坐标
        x, y = self.calc_mov_point(angle)
        self.click(x, y, t)

    def calc_move_point_direction(self, direction: str):
        if direction is None:
            return None
        # 计算轮盘x, y坐标
        angle = 0
        if direction == 'up':
            angle = 90
        if direction == 'down':
            angle = 270
        if direction == 'left':
            angle = 180
        x, y = self.calc_mov_point(angle)
        return x, y

    def juexing(self):
        # 往下走两步，放觉醒
        self.move(350, 0.1)
        self.move(270, 0.1)
        self.skill_r()
        time.sleep(1)
        self.skill_r()

    def continuous_attack(self, index: int):
        index = index % 6
        print(f"开始放连招：第{index}套。。")
        self.skill_down()
        self.skill_q()
        self.skill_right()
        if index == 0:
            self.skill_w()
            self.skill_1()
            self.attack(3)
            self.skill_1()
            self.attack()
            self.skill_1()
            self.skill_t()
            self.skill_t()
        if index == 1:
            self.skill_2()
            self.attack()
            self.skill_2()
            self.attack()
            self.skill_2()
            self.skill_2()
            self.attack()
        if index == 2:
            self.skill_d()
            self.attack(2)
            self.skill_3()
            time.sleep(0.2)
            self.skill_3()
            self.attack(3)
        if index == 3:
            self.skill_4()
            time.sleep(0.5)
            self.skill_y()
            self.skill_y()
            self.attack(3)
            self.skill_4()
            self.attack(3)
        if index == 4:
            self.skill_5()
            time.sleep(0.2)
            self.skill_5()
            self.attack(3)
            self.skill_e()
            self.attack(3)
        else:
            self.attack(3)
            self.skill_right()
            time.sleep(0.1)
            self.skill_up()
            time.sleep(0.1)
            self.skill_down()
            time.sleep(0.1)
            self.skill_left()


    def attack(self, cnt: int = 1):
        x, y = (1962, 910)
        for i in range(cnt):
            self.click(x, y)
            time.sleep(0.1)


    def skill_d(self, t: float = 0.01):
        x, y = (1325, 982)
        self.click(x, y, t)

    def skill_f(self, t: float = 0.01):
        x, y = (1488, 967)
        self.click(x, y, t)

    def skill_1(self, t: float = 0.01):
        x, y = (1636, 961)
        self.click(x, y, t)

    def skill_2(self, t: float = 0.01):
        x, y = (1695, 818)
        self.click(x, y, t)

    def skill_3(self, t: float = 0.01):
        x, y = (1807, 658)
        self.click(x, y, t)

    def skill_4(self, t: float = 0.01):
        x, y = (1980, 652)
        self.click(x, y, t)

    def skill_5(self, t: float = 0.01):
        x, y = (1982, 492)
        self.click(x, y, t)

    def skill_t(self, t: float = 0.01):
        x, y = (1792, 985)
        self.click(x, y, t)

    def skill_y(self, t: float = 0.01):
        x, y = (2018, 780)
        self.click(x, y, t)

    def skill_q(self, t: float = 0.01):
        x, y = (1661, 330)
        self.click(x, y, t)

    def skill_w(self, t: float = 0.01):
        x, y = (1774, 328)
        self.click(x, y, t)

    def skill_e(self, t: float = 0.01):
        x, y = (1862, 330)
        self.click(x, y, t)

    def skill_r(self, t: float = 0.01):
        x, y = (1979, 330)
        self.click(x, y, t)

    def skill_up(self, t: float = 0.1):
        x, y = (1812, 518)
        x, y = self._ramdon_xy(x, y)
        self.adb.slow_swipe(x, y, x, y - 100, duration=t, steps=1)

    def skill_down(self, t: float = 0.1):
        x, y = (1812, 518)
        x, y = self._ramdon_xy(x, y)
        self.adb.slow_swipe(x, y, x, y + 100, duration=t, steps=1)

    def skill_left(self, t: float = 0.1):
        x, y = (1812, 518)
        x, y = self._ramdon_xy(x, y)
        self.adb.slow_swipe(x, y, x - 100, y, duration=t, steps=1)

    def skill_right(self, t: float = 0.1):
        x, y = (1812, 518)
        x, y = self._ramdon_xy(x, y)
        self.adb.slow_swipe(x, y, x + 100, y, duration=t, steps=1)

    def click(self, x, y, t: float = 0.01):
        x, y = self._ramdon_xy(x, y)
        self.adb.touch_start(x, y)
        time.sleep(t)
        self.adb.touch_end(x, y)

    def _ramdon_xy(self, x, y):
        x = x + random.randint(-5, 5)
        y = y + random.randint(-5, 5)
        return x, y

if __name__ == '__main__':
    ctl = GameControl(ScrcpyADB())
    ctl.skill_left()
    # ctl.move(180, 3)
    # time.sleep(0.3)
    # ctl.attack()
    # time.sleep(0.3)
    # ctl.move(270, 5)
    # time.sleep(0.3)
    # ctl.attack(3)


