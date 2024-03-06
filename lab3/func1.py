#Functions1
#ex1
def grams_to_ounces(grams):
    ounces = 28.3495231 * grams
    return ounces

# str = input()
# n = int(str)
# print(grams_to_ounces(n))

#ex2
def fahrenheit_to_celsius(fahrenheit):
    celsius = (5 / 9) * (fahrenheit - 32)
    return celsius

# str = input()
# n = int(str)
# print(fahrenheit_to_celsius(n))

#ex3
def solve(numheads, numlegs):
    for chickens in range(numheads + 1):
        rabbits = numheads - chickens
        if (2 * chickens + 4 * rabbits) == numlegs:
            return chickens, rabbits
    return None

# numheads1 = input()
# numlegs1 = input()
# numheads2 = int(numheads1)
# numlegs2 = int(numlegs1)
# print(solve(numheads2, numlegs2))

#ex4
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def filter_prime(numbers):
    return [num for num in numbers if is_prime(num)]

# n = input(int())
# print(filter_prime(n))

#ex5
from itertools import permutations

def string_permutations(input_str):
    return [''.join(p) for p in permutations(input_str)]

# str = input()
# print(string_permutations(str))

#ex6
def reverse_words(input_str):
    words = input_str.split()
    reversed_sentence = ' '.join(reversed(words))
    return reversed_sentence

# str = input()
# print(reverse_words(str))

#ex7
def has_33(nums):
    for i in range(len(nums) - 1):
        if nums[i] == 3 and nums[i + 1] == 3:
            return True
    return False

# str = input()
# list = str.split()
# list = [int(item) for item in list]
# print(has_33(list))

#ex8
def spy_game(nums):
    code = [0, 0, 7, 'x']
    for num in nums:
        if num == code[0]:
            code.pop(0)
    return len(code) == 1

# str = input()
# list = str.split()
# list = [int(item) for item in list]
# print(spy_game(list))

#ex9
def sphere_volume(radius):
    volume = (4 / 3) * 3.141592653589793 * (radius**3)
    return volume

# radius = int(input())
# print(sphere_volume(radius))

#ex10
def unique_elements(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

# nums = input()
# print(unique_elements(nums))

#ex11
def is_palindrome(word):
    return word == word[::-1]

# word = input()
# print(is_palindrome(word))

#ex12
def histogram(numbers):
    for num in numbers:
        print('*' * num)

# list = []
# str = input()
# list = str.split(" ")
# list = [int(i) for i in list]
# print(histogram(list))

#ex13
import random

def guess_the_number():
    print("Hello! What is your name?")
    name = input()
    print(f"Well, {name}, I am thinking of a number between 1 and 20.")
    
    num = random.randint(1, 20)
    tries = 0

    while True:
        print("Take a guess.")
        guess = int(input())
        tries += 1

        if guess < num:
            print("Your guess is too low.")
        elif guess > num:
            print("Your guess is too high.")
        else:
            print(f"Good job, {name}! You guessed my number in {tries} guesses!")
            break

# print(guess_the_number())



#Functions2
        
# Dictionary of movies
movies = [
{
"name": "Usual Suspects", 
"imdb": 7.0,
"category": "Thriller"
},
{
"name": "Hitman",
"imdb": 6.3,
"category": "Action"
},
{
"name": "Dark Knight",
"imdb": 9.0,
"category": "Adventure"
},
{
"name": "The Help",
"imdb": 8.0,
"category": "Drama"
},
{
"name": "The Choice",
"imdb": 6.2,
"category": "Romance"
},
{
"name": "Colonia",
"imdb": 7.4,
"category": "Romance"
},
{
"name": "Love",
"imdb": 6.0,
"category": "Romance"
},
{
"name": "Bride Wars",
"imdb": 5.4,
"category": "Romance"
},
{
"name": "AlphaJet",
"imdb": 3.2,
"category": "War"
},
{
"name": "Ringing Crime",
"imdb": 4.0,
"category": "Crime"
},
{
"name": "Joking muck",
"imdb": 7.2,
"category": "Comedy"
},
{
"name": "What is the name",
"imdb": 9.2,
"category": "Suspense"
},
{
"name": "Detective",
"imdb": 7.0,
"category": "Suspense"
},
{
"name": "Exam",
"imdb": 4.2,
"category": "Thriller"
},
{
"name": "We Two",
"imdb": 7.2,
"category": "Romance"
}
]
#ex1
def is_high_score(i):
    return i["imdb"] > 5.5

#ex2
def high_score_movies(movies):
    return [i["name"] for i in movies if is_high_score(i)]

#ex3
def movies_by_category(movies, category):
    return [i["name"] for i in movies if i["category"] == category]

#ex4
def average_imdb_score(movies):
    total_score = sum(movie["imdb"] for movie in movies)
    return total_score / len(movies)

#ex5
def average_imdb_score_by_category(category):
    total = 0
    new = []
    for movie in movies:
        if movie["category"] == category:
            new.append(movie["imdb"])
    for i in new:
        total += i
    average = total / len(new)    
    return average

# Test the functions
# print("Answer to movies ex1: ", is_high_score(movies[13]))
# print("Answer to movies ex2: ", high_score_movies(movies))
# print("Answer to movies ex3: ", movies_by_category(movies, "Romance"))
# print("Answer to movies ex4: ", average_imdb_score(movies))
# print("Answer to movies ex5: ", average_imdb_score_by_category(category = input()))
