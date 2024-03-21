#ex1
def generators(num):
    n = 1
    while n <= num:
        yield n ** 2
        n += 1
num = int(input("Give me a number "))
for n in generators(num):
    print(n)

#ex2
def even(n):
    iter = 1
    for i in range(n + 1):
        if(i % 2 == 0):
            yield i
            iter += 1
n = int(input("Give me a number "))
for n in even(n):
    print(n)

#ex3
def divisible(n):
    iter = 1
    for i in range(n+1):
        if(i % 3 == 0 and i % 4 == 0):
            yield i
            iter += 1
n = int(input("Give me a number "))
for n in divisible(n):
    print(n)

#ex4
def squares(a, b):
    iter = 1
    for i in range(a, b + 1):
        yield i ** 2
        iter += 1
a = int(input("Give me the second number "))
b = int(input("Give me the second number "))
for i in squares(a,b):
    print(i)

#ex5
def down(n):
    iter = 1
    for i in range(n, -1, -1):
        yield i
        iter += 1
n = int(input("Give me a number "))
for i in down(n):
    print(i)