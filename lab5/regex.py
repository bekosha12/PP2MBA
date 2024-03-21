import re

file = open('row.txt', 'r', encoding = 'utf-8')
# print(file)
s = input()

#ex1
reg1 = re.compile('ab*')
result1 = reg1.findall(s)
print(result1)

#ex2
reg2 = re.compile('ab{2,3}')
result2 = reg2.findall(s)
print(result2)

#ex3
reg3 = r'\b[a-z]+_[a-z]+\b'
matches = re.findall(reg3, s)
for i in matches:
    print(i)

#ex4
reg4 = r'\b[A-Z][a-z]+\b'
matches = re.findall(reg4, s)
for i in matches:
    print(i)

#ex5
reg5 = r'a.*b$'
matches = re.findall(reg5, s)
for i in matches:
    print(i)

#ex6
reg6 = r'[ ,.]'
repl = re.sub(reg6, ':', s)
print(repl)

#ex7
words = s.split('_')
camel_case_string = words[0] + ''.join(word.capitalize() for word in words[1:])
print(camel_case_string)

#ex8
reg8 = r'[A-Z][^A-Z]*'
matches = re.findall(reg8, s)
for i in matches:
    print(i)

#ex9
def capital(s):
    return re.sub(r"(\w)([A-Z])", r"\1 \2", s)
result = capital(s)
print(result)

#ex10
def camel_to_snake(str):
        str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str).lower()
result = camel_to_snake(s)
print(result)
