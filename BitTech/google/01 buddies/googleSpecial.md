### get the buddies from list
### inside that list there are strings like "aaa", "zzz","abc","efg","acfh","kmoq"
### here need to find buddy who has same length and same distance from of their characters in python


## Issues with your approach:

1. **Distance calculation is unclear** - You mentioned "distance between characters" but didn't specify if it's adjacent character differences or something else
2. **Dictionary structure is confusing** - Using the string as key with distance list as value makes comparison inefficient
3. **Matching logic is incomplete** - How do you efficiently find strings with identical distance patterns?

## Problem clarification:
I assume "same distance" means the differences between consecutive characters are identical. For example:
- "abc" has distances [1, 1] (b-a=1, c-b=1)
- "def" has distances [1, 1] (e-d=1, f-e=1)
- These would be buddies

## Solutions:## Analysis:

**Brute Force Solution: O(n²)**
- Compare every pair of strings
- Simple but inefficient for large inputs
- Time complexity: O(n² × m) where m is average string length

**Optimal Solution: O(n)**
- Group strings by their pattern signature
- Use hash map for O(1) average lookup
- Time complexity: O(n × m) where m is average string length
- Space complexity: O(n × m)

**Your approach was bad because:**
1. You stored individual strings as keys, making comparison O(n²)
2. The distance calculation wasn't clearly defined
3. No efficient way to find matching patterns

**Key insights:**
1. Use the distance pattern as the grouping key, not individual strings
2. Convert distance list to tuple for hashing
3. Group first, then extract pairs - more efficient than comparing everything

The optimal solution is much better for large datasets and is what most interviewers would expect after the brute force approach.


def get_char_distances(s):
    """Get distances between consecutive characters"""
    if len(s) < 2:
        return []
    return [ord(s[i+1]) - ord(s[i]) for i in range(len(s)-1)]



# BRUTE FORCE SOLUTION - O(n²)
def find_buddies_brute_force(strings):
    """Compare every pair of strings"""
    buddies = []
    n = len(strings)
    
    for i in range(n):
        for j in range(i+1, n):
            s1, s2 = strings[i], strings[j]
            
            # Same length check
            if len(s1) == len(s2):
                # Same distance pattern check
                if get_char_distances(s1) == get_char_distances(s2):
                    buddies.append((s1, s2))
    
    return buddies




# OPTIMAL SOLUTION - O(n) average case
def find_buddies_optimal(strings):
    """Group strings by their distance pattern"""
    from collections import defaultdict
    
    # Group by (length, distance_pattern)
    pattern_groups = defaultdict(list)
    
    for s in strings:
        length = len(s)
        distances = tuple(get_char_distances(s))  # tuple for hashing
        key = (length, distances)
        pattern_groups[key].append(s)
    
    # Extract buddy pairs
    buddies = []
    for group in pattern_groups.values():
        if len(group) > 1:
            # Generate all pairs within the group
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    buddies.append((group[i], group[j]))
    
    return buddies




# EVEN MORE OPTIMAL - Return groups instead of pairs
def find_buddy_groups(strings):
    """Return groups of buddy strings"""
    from collections import defaultdict
    
    pattern_groups = defaultdict(list)
    
    for s in strings:
        length = len(s)
        distances = tuple(get_char_distances(s))
        key = (length, distances)
        pattern_groups[key].append(s)
    
    # Return only groups with multiple strings
    return [group for group in pattern_groups.values() if len(group) > 1]




# Test the solutions
if __name__ == "__main__":
    test_strings = ["aaa", "zzz", "abc", "efg", "acfh", "kmoq", "def", "bbb"]
    
    print("Test strings:", test_strings)
    print()
    
    # Show distance patterns
    print("Distance patterns:")
    for s in test_strings:
        print(f"'{s}' -> {get_char_distances(s)}")
    print()
    
    # Brute force solution
    print("Brute Force Solution:")
    buddies_bf = find_buddies_brute_force(test_strings)
    for pair in buddies_bf:
        print(f"  {pair}")
    print()
    
    # Optimal solution
    print("Optimal Solution:")
    buddies_opt = find_buddies_optimal(test_strings)
    for pair in buddies_opt:
        print(f"  {pair}")
    print()
    
    # Group solution
    print("Buddy Groups:")
    groups = find_buddy_groups(test_strings)
    for i, group in enumerate(groups, 1):
        print(f"  Group {i}: {group}")

