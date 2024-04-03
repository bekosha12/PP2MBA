import os

#ex1
def list_contents(path):
    print("Directories:")
    for dir_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir_name)):
            print(dir_name)

    print("\nFiles:")
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            print(file_name)

path = input("Enter the path: ")
list_contents(path)

#ex2
def check_access(path):
    print(f"Existence: {os.path.exists(path)}")
    print(f"Readable: {os.access(path, os.R_OK)}")
    print(f"Writable: {os.access(path, os.W_OK)}")
    print(f"Executable: {os.access(path, os.X_OK)}")

path = input("Enter the path: ")
check_access(path)

#ex3
def path_info(path):
    if os.path.exists(path):
        print("Path exists")
        print("Filename:", os.path.basename(path))
        print("Directory portion:", os.path.dirname(path))
    else:
        print("Path does not exist")

path = input("Enter the path: ")
path_info(path)

#ex4
def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)

file_path = input("Enter the file path: ")
print("Number of lines:", count_lines(file_path))

#ex5
def write_list_to_file(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

file_path = input("Enter the file path: ")
data = input("Enter list elements separated by space: ").split()
write_list_to_file(file_path, data)

#ex6
import string

for letter in string.ascii_uppercase:
    with open(letter + ".txt", 'w') as file:
        pass

#ex7
def copy_file(source, destination):
    with open(source, 'r') as src_file:
        with open(destination, 'w') as dest_file:
            for line in src_file:
                dest_file.write(line)

source = input("Enter the source file path: ")
destination = input("Enter the destination file path: ")
copy_file(source, destination)

#ex8
def delete_file(file_path):
    if os.path.exists(file_path):
        if os.access(file_path, os.W_OK):
            os.remove(file_path)
            print("File deleted successfully")
        else:
            print("You don't have permission to delete this file")
    else:
        print("File does not exist")

file_path = input("Enter the file path to delete: ")
delete_file(file_path)
