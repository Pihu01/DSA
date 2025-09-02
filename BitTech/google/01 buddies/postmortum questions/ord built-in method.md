`ord()` is a built-in Python function that returns the **Unicode code point** (integer value) of a single character.

## Basic Usage:
```python
ord('a')  # Returns 97
ord('b')  # Returns 98
ord('A')  # Returns 65
ord('1')  # Returns 49
ord(' ')  # Returns 32 (space)
```

## Why we use it for character distances:

When I used `ord(s[i+1]) - ord(s[i])` in the buddy strings code, I was calculating the **numeric difference** between characters:

```python
# Example with "abc"
ord('b') - ord('a')  # 98 - 97 = 1
ord('c') - ord('b')  # 99 - 98 = 1
# So "abc" has distance pattern [1, 1]

# Example with "ace" 
ord('c') - ord('a')  # 99 - 97 = 2
ord('e') - ord('c')  # 101 - 99 = 2  
# So "ace" has distance pattern [2, 2]
```

## The opposite function is `chr()`:
```python
chr(97)   # Returns 'a'
chr(98)   # Returns 'b'
chr(65)   # Returns 'A'
```

## ASCII values for common characters:
- 'a' to 'z': 97 to 122
- 'A' to 'Z': 65 to 90  
- '0' to '9': 48 to 57
- Space ' ': 32

This is why `ord('b') - ord('a') = 1` - consecutive letters in the alphabet have consecutive Unicode values, making it perfect for measuring "distance" between characters in our buddy strings problem.
