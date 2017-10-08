# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:02:59 2017

@author: Toshiharu
"""

class suma():
    def __init__(self):
        self.a = 0
        self.b = 0
        self.abc = 0
        
    def resultado(self):
        self.abc = self.a+self.b
    
test = suma()
test.a = 4
#test.b = 5
test.resultado()
print(test.abc)