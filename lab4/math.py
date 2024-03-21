import math

#ex1
degrees = float(input("Enter the degree: "))
radians = degrees * (math.pi / 180)
print(round(radians, 6))

#ex2
h = float(input("Height: "))
b1 = float(input("Base, first value: "))
b2 = float(input("Base, second value: "))
area = 0.5 * (b1 + b2) * h
print("Expected Output: ", area)

#ex3
n = int(input("Input number of sides: "))
s = int(input("Input the length of a side: "))
area = (n * s ** 2) / (4 * math.tan(math.pi / n))
print("The area of the polygon is: ", int(area))

#ex4
l = float(input("Length of base: "))
h = float(input("Height of parallelogram: "))
area = l * h
print("Expected Output: ", area)