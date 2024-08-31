import numpy as np


class Linear:
    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.m = (self.y1-self.y0)/(self.x1-self.x0)
        
    def __call__(self, x):
        return self.y0 + x*self.m
    
    def diff(self, x):
        return self.m