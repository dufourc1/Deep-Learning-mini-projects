from sys import argv, exit
import torch

def foo():
    print("This is a test")

tests = {'test': foo}

if __name__ == '__main__':

    if len(argv) < 2:
        print("A model has to be chosen")
        exit(1)
    if tests.get(argv[1].lower()) is None:
        print("Invalid model")
        exit(1)

    test = tests.get(argv[1].lower())

    test()
