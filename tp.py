def is_palindrome(num):
        return str(num) == str(num)[::-1]
    
    # Test the function
num = 12321
print(is_palindrome(num))