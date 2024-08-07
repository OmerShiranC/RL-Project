import json
import pickle


def get_valid_input(a, b):
    """
    Get a valid integer input from the user between a and b
    args:
        param a: lower bound
        param b: upper bound
    return:
        integer between a and b or 'q' to quit
    """
    while True:
        user_input = input(f"Enter an integer between {a} and {b}, or 'q' to quit: ").strip().lower()

        if user_input == 'q':
            return 'q'

        try:
            number = int(user_input)
            if a <= number <= b:
                return number
            else:
                print(f"Please enter a number between {a} and {b}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer or 'q'.")

