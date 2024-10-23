#!/usr/bin/python
# -*- coding: utf-8 -*-

# This file was created using the DirectGUI Designer

from direct.gui import DirectGuiGlobals as DGG

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import (
    LPoint3f,
    LVecBase3f,
    LVecBase4f,
    TextNode
)


class GUI:
    def __init__(self, rootParent=None):
        # 1.85
        self.pg234 = DirectFrame(
            frameSize=(0.0, 1, -2.0, 0.0),
            frameColor=(1.0, 0.0, 1.0, 1.0),
            pos=LPoint3f(1.4, 0, 1),
            parent=rootParent,
        )
        self.pg234.setTransparency(0)

    def show(self):
        self.pg234.show()

    def hide(self):
        self.pg234.hide()

    def destroy(self):
        self.pg234.destroy()


if __name__ == "__main__":
    from huri.core.common_import import wd

    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    GUI()
    base.run()
