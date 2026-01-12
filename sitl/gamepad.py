import pygame
from copy import copy
import numpy as np
import asyncio
import json


betaflight_order = ["roll", "pitch", "throttle", "yaw", "arm"] # AETR

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    joystick = None


class Gamepad:
    def __init__(self, mapping, callback):
        if mapping == 'edgetx':
            self.gamepad_mapping = 'edgetx'
        else:
            with open(mapping, 'r') as f:
                self.gamepad_mapping = json.load(f)
        self.callback = callback

    async def run(self):
        while True:
            await asyncio.sleep(0.01)
            pygame.event.pump()
            if joystick is None:
                continue
            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            if self.gamepad_mapping == 'edgetx':
                rc = [int((v + 1) * 500 + 1000) for v in axes]
            else:
                rc = []
                for key in betaflight_order:
                    cfg = self.gamepad_mapping[key]
                    if "axis" in cfg:
                        idx = cfg["axis"]
                        v = axes[idx] if idx < len(axes) else 0.0
                    else:
                        idx = cfg["button"]
                        v = 1.0 if idx < len(buttons) and buttons[idx] else 0.0
                    if cfg.get("invert", False):
                        v = -v
                    rc.append(int((v + 1) * 500 + 1000))
            print(rc)
            self.callback(rc)

def main():
    initial_axes = None
    while True:
        pygame.event.pump()
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        if initial_axes is None:
            initial_axes = copy(axes)
        if sum(np.abs(np.array(axes) - initial_axes) > 0.25) == 1:
            diffs = np.abs(np.array(axes) - initial_axes)
            print("Axis: ", np.argmax(diffs), "Value: ", axes[np.argmax(diffs)])
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        for i, button in enumerate(buttons):
            if button:
                print(f"Button {i} pressed")


if __name__ == "__main__":
    main()