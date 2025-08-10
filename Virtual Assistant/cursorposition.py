import pyautogui
import time 

while True:
    x,y=pyautogui.position()
    print(f"The current position of cursor is {x,y}")
    time.sleep(2)