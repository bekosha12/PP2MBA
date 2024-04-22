# import mymodule as mm

# a = mm.person1["country"]
# print(a)

# import platform

# x = platform.system()
# y = dir(platform)
# print(x, y)

# from mymodule import person1

# print(person1["age"])

import datetime

x = datetime.datetime.now()
print(x)

a = datetime.datetime(2020, 5, 17)
print(a.strftime("%C"))