import numpy as np

# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
class narcissusAttack(object):

    def __init__(self, trigger, multiplier):
        self.trigger = trigger
        self.multiplier = multiplier

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        img = img.astype(np.int16)
        img += self.trigger * self.multiplier
        return img
