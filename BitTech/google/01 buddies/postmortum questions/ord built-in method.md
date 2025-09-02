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

------------------------------------------------------------------
If you want to calculate character distances **manually** without using `ord()`, you can create your own character-to-number mapping. Here are a few approaches:## Manual approaches explained:

**Method 1: Alphabet string indexing**
- Use `"abcdefghijklmnopqrstuvwxyz".index(char)` 
- 'a' → 0, 'b' → 1, 'c' → 2, etc.
- Simple but `index()` is O(26) for each lookup

**Method 2: Dictionary mapping** 
- Pre-create a dictionary mapping each character to its position
- O(1) lookup time, more efficient
- Best balance of manual + performance

**Method 3: Hardcoded mapping**
- Completely manual - write out every character mapping
- Most "manual" but also most verbose

## Why you might want manual approach:
- Interview constraint (no built-in functions)
- Educational purposes 
- Working with non-standard character sets
- Language doesn't have `ord()` equivalent

## Performance comparison:
1. `ord()` - Fastest (built-in C function)
2. Dictionary lookup - Fast (O(1))
3. Alphabet indexing - Slower (O(26) per lookup)
4. Hardcoded - Fast but impractical for large character sets

For interviews, I'd recommend the **dictionary approach** - it shows you understand the concept while being efficient!

# APPROACH 1: Using alphabet string index
def char_to_num_alphabet(char):
    """Convert a-z to 0-25 using string index"""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return alphabet.index(char.lower())

def get_distances_manual_v1(s):
    """Calculate distances using alphabet indexing"""
    if len(s) < 2:
        return []
    
    distances = []
    for i in range(len(s) - 1):
        num1 = char_to_num_alphabet(s[i])
        num2 = char_to_num_alphabet(s[i + 1])
        distances.append(num2 - num1)
    
    return distances

# APPROACH 2: Using dictionary mapping
def create_char_map():
    """Create character to number mapping"""
    char_map = {}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i, char in enumerate(alphabet):
        char_map[char] = i
    return char_map

def get_distances_manual_v2(s, char_map):
    """Calculate distances using dictionary lookup"""
    if len(s) < 2:
        return []
    
    distances = []
    for i in range(len(s) - 1):
        num1 = char_map[s[i].lower()]
        num2 = char_map[s[i + 1].lower()]
        distances.append(num2 - num1)
    
    return distances

# APPROACH 3: Hardcoded mapping (most manual)
def char_to_num_hardcoded(char):
    """Hardcoded character mapping"""
    mapping = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8,
        'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16,
        'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
    }
    return mapping[char.lower()]

def get_distances_manual_v3(s):
    """Calculate distances using hardcoded mapping"""
    if len(s) < 2:
        return []
    
    distances = []
    for i in range(len(s) - 1):
        num1 = char_to_num_hardcoded(s[i])
        num2 = char_to_num_hardcoded(s[i + 1])
        distances.append(num2 - num1)
    
    return distances

# Complete buddy finder using manual approach
def find_buddies_manual(strings):
    """Find buddies without using ord()"""
    from collections import defaultdict
    
    # Create character mapping once
    char_map = create_char_map()
    pattern_groups = defaultdict(list)
    
    for s in strings:
        length = len(s)
        distances = tuple(get_distances_manual_v2(s, char_map))
        key = (length, distances)
        pattern_groups[key].append(s)
    
    # Return buddy groups
    return [group for group in pattern_groups.values() if len(group) > 1]

# Test all approaches
if __name__ == "__main__":
    test_strings = ["abc", "def", "xyz", "ace", "bdf"]
    
    print("Testing manual approaches:")
    print("Strings:", test_strings)
    print()
    
    # Test each approach
    for s in test_strings:
        dist1 = get_distances_manual_v1(s)
        
        char_map = create_char_map()
        dist2 = get_distances_manual_v2(s, char_map)
        
        dist3 = get_distances_manual_v3(s)
        
        print(f"'{s}':")
        print(f"  Method 1 (alphabet.index): {dist1}")
        print(f"  Method 2 (dictionary):     {dist2}")
        print(f"  Method 3 (hardcoded):      {dist3}")
        print()
    
    # Find buddies manually
    print("Buddy groups (manual method):")
    groups = find_buddies_manual(test_strings)
    for i, group in enumerate(groups, 1):
        print(f"  Group {i}: {group}")
    
    # Show step-by-step example
    print("\nStep-by-step example for 'abc':")
    s = "abc"
    print(f"String: '{s}'")
    print(f"'a' -> position 0")
    print(f"'b' -> position 1") 
    print(f"'c' -> position 2")
    print(f"Distance 'b'-'a' = 1-0 = 1")
    print(f"Distance 'c'-'b' = 2-1 = 1")
    print(f"Final pattern: [1, 1]")
