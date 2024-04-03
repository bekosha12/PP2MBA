#ex1
def multiply_list(numbers):
    import functools
    return functools.reduce(lambda x, y: x * y, numbers)

numbers = list(map(int, input("Enter numbers: ").split()))
result = multiply_list(numbers)
print("Result:", result)

#ex2
def count_case_letters(string):
    upper_count = sum(1 for char in string if char.isupper())
    lower_count = sum(1 for char in string if char.islower())
    return upper_count, lower_count

string = input("Enter a string: ")

upper, lower = count_case_letters(string)
print("Number of upper case letters:", upper)
print("Number of lower case letters:", lower)

#ex3
def is_palindrome(string):
    return string == string[::-1]

string = input("Enter a string: ")
if is_palindrome(string):
    print("The string is a palindrome.")
else:
    print("The string is not a palindrome.")

#ex4
import time
import math

def square_root_after_milliseconds(number, milliseconds):
    time.sleep(milliseconds / 1000)
    result = math.sqrt(number)
    print(f"Square root of {number} after {milliseconds} milliseconds is {result}")

number = int(input("Enter a number: "))
milliseconds = int(input("Enter milliseconds: "))

square_root_after_milliseconds(number, milliseconds)

#ex5
def all_true(elements):
    return all(elements)

elements = tuple(map(bool, input("Enter elements separated by space: ").split()))

result = all_true(elements)
print("All elements are True:", result)
