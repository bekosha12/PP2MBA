# def greeting(name):
#   print("Hello, " + name)

# person1 = {
#   "name": "John",
#   "age": 36,
#   "country": "Norway"
# }

import math
import re

# def volume_cyl(r, h):
#   vol = math.pi*h*(r**2)
#   return vol

# h = 5
# r = 4

# print(volume_cyl(h, r))

s = "The_Alps_are*!mountains?"
x = re.sub("[_?!*]", " ", s)

print(x)