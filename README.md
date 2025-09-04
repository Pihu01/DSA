# Complete DSA Patterns Guide with Python Examples

## 1. Two Pointers Pattern

**When to use:** When you need to find pairs, compare elements, or work with sorted arrays.

**Key Idea:** Use two pointers moving towards each other or in the same direction.

```python
# Example: Two Sum in Sorted Array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

# Example: Remove Duplicates from Sorted Array
def remove_duplicates(nums):
    if not nums:
        return 0
    
    write_index = 1
    for read_index in range(1, len(nums)):
        if nums[read_index] != nums[read_index - 1]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index
```

## 2. Sliding Window Pattern

**When to use:** For problems involving contiguous subarrays/substrings with specific conditions.

**Key Idea:** Maintain a window and expand/contract it based on conditions.

```python
# Example: Maximum Sum Subarray of Size K
def max_sum_subarray_k(nums, k):
    if len(nums) < k:
        return -1
    
    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example: Longest Substring Without Repeating Characters
def longest_unique_substring(s):
    char_map = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

## 3. Fast & Slow Pointers (Floyd's Cycle Detection)

**When to use:** For detecting cycles in linked lists or arrays.

**Key Idea:** Use two pointers moving at different speeds.

```python
# Example: Detect Cycle in Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Example: Find Middle of Linked List
def find_middle(head):
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

## 4. Merge Intervals Pattern

**When to use:** For problems involving overlapping intervals.

**Key Idea:** Sort intervals and merge overlapping ones.

```python
# Example: Merge Overlapping Intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        if current[0] <= last_merged[1]:  # Overlapping
            merged[-1] = [last_merged[0], max(last_merged[1], current[1])]
        else:
            merged.append(current)
    
    return merged

# Example: Insert Interval
def insert_interval(intervals, new_interval):
    result = []
    i = 0
    
    # Add all intervals before the new interval
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    # Add remaining intervals
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    
    return result
```

## 5. Cyclic Sort Pattern

**When to use:** For problems with numbers in a given range (usually 1 to n).

**Key Idea:** Place each number at its correct index.

```python
# Example: Find Missing Number
def find_missing_number(nums):
    i = 0
    while i < len(nums):
        if nums[i] < len(nums) and nums[i] != nums[nums[i]]:
            # Swap nums[i] with nums[nums[i]]
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        else:
            i += 1
    
    # Find the missing number
    for i in range(len(nums)):
        if nums[i] != i:
            return i
    
    return len(nums)

# Example: Find All Duplicates
def find_duplicates(nums):
    i = 0
    while i < len(nums):
        if nums[i] != nums[nums[i] - 1]:
            # Swap to correct position
            correct_pos = nums[i] - 1
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    
    duplicates = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            duplicates.append(nums[i])
    
    return duplicates
```

## 6. Tree Traversal Patterns

**When to use:** For tree-based problems.

**Key Idea:** Use DFS or BFS based on the problem requirements.

```python
# Tree Node Definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# DFS - Inorder Traversal
def inorder_traversal(root):
    result = []
    
    def dfs(node):
        if node:
            dfs(node.left)
            result.append(node.val)
            dfs(node.right)
    
    dfs(root)
    return result

# BFS - Level Order Traversal
from collections import deque

def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result

# Example: Path Sum
def has_path_sum(root, target_sum):
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == target_sum
    
    remaining_sum = target_sum - root.val
    return (has_path_sum(root.left, remaining_sum) or 
            has_path_sum(root.right, remaining_sum))
```

## 7. Binary Search Pattern

**When to use:** For sorted arrays or when you need to find a specific condition.

**Key Idea:** Eliminate half of the search space in each iteration.

```python
# Standard Binary Search
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Find First Occurrence
def find_first_occurrence(nums, target):
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Search in Rotated Sorted Array
def search_rotated_array(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

## 8. Top K Elements Pattern

**When to use:** When you need to find the top/bottom K elements.

**Key Idea:** Use heaps, quickselect, or sorting.

```python
import heapq

# Example: K Largest Elements
def find_k_largest(nums, k):
    # Use min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return list(heap)

# Example: Kth Largest Element
def find_kth_largest(nums, k):
    # Use max heap (negate values for min heap)
    heap = [-num for num in nums]
    heapq.heapify(heap)
    
    for _ in range(k - 1):
        heapq.heappop(heap)
    
    return -heapq.heappop(heap)

# Example: Top K Frequent Elements
def top_k_frequent(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # Use heap with frequency as key
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]
```

## 9. K-way Merge Pattern

**When to use:** When you need to merge K sorted arrays/lists.

**Key Idea:** Use a min heap to keep track of the smallest elements.

```python
import heapq

# Example: Merge K Sorted Lists
def merge_k_sorted_lists(lists):
    heap = []
    
    # Add first element of each list to heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from the same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

# Example: Smallest Range Covering Elements from K Lists
def smallest_range(nums):
    heap = []
    max_val = float('-inf')
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(nums):
        heapq.heappush(heap, (lst[0], i, 0))
        max_val = max(max_val, lst[0])
    
    range_start, range_end = 0, float('inf')
    
    while heap:
        min_val, list_idx, elem_idx = heapq.heappop(heap)
        
        # Update range if current range is smaller
        if max_val - min_val < range_end - range_start:
            range_start, range_end = min_val, max_val
        
        # Add next element from the same list
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            max_val = max(max_val, next_val)
        else:
            break
    
    return [range_start, range_end]
```

## 10. Dynamic Programming Patterns

**When to use:** For optimization problems with overlapping subproblems.

**Key Idea:** Break down problem into smaller subproblems and store results.

```python
# Example: Fibonacci with Memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Example: Coin Change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Example: Longest Common Subsequence
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Example: 0/1 Knapsack
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],  # Don't include current item
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]  # Include current item
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]
```

## 11. Backtracking Pattern

**When to use:** For generating all possible solutions or finding valid combinations.

**Key Idea:** Try all possibilities and backtrack when a path doesn't work.

```python
# Example: Generate All Subsets
def generate_subsets(nums):
    result = []
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])  # Add current subset
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Backtrack
    
    backtrack(0, [])
    return result

# Example: N-Queens
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check anti-diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def solve(board, row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                solve(board, row + 1)
                board[row][col] = '.'
    
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    solve(board, 0)
    return result

# Example: Combination Sum
def combination_sum(candidates, target):
    result = []
    
    def backtrack(start, current_combination, remaining_target):
        if remaining_target == 0:
            result.append(current_combination[:])
            return
        
        if remaining_target < 0:
            return
        
        for i in range(start, len(candidates)):
            current_combination.append(candidates[i])
            backtrack(i, current_combination, remaining_target - candidates[i])
            current_combination.pop()
    
    backtrack(0, [], target)
    return result
```

## 12. Graph Traversal Patterns

**When to use:** For problems involving graphs, trees, or connected components.

**Key Idea:** Use DFS or BFS to explore the graph.

```python
from collections import deque, defaultdict

# Example: DFS
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Example: BFS
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example: Number of Islands
def num_islands(grid):
    if not grid:
        return 0
    
    def dfs(i, j):
        if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or 
            grid[i][j] == '0'):
            return
        
        grid[i][j] = '0'  # Mark as visited
        
        # Explore all 4 directions
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    
    return count

# Example: Course Schedule (Topological Sort)
def can_finish(num_courses, prerequisites):
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Find courses with no prerequisites
    queue = deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)
    
    completed = 0
    while queue:
        course = queue.popleft()
        completed += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return completed == num_courses
```

## Pattern Recognition Tips

1. **Two Pointers**: Look for problems with sorted arrays, palindromes, or pair finding
2. **Sliding Window**: Contiguous subarray/substring problems with constraints
3. **Fast & Slow Pointers**: Cycle detection, finding middle elements
4. **Merge Intervals**: Overlapping intervals, scheduling problems
5. **Cyclic Sort**: Numbers in range [1, n], finding missing/duplicate numbers
6. **Tree Traversal**: Any tree-related problem
7. **Binary Search**: Sorted arrays, finding specific conditions
8. **Top K Elements**: Finding largest/smallest K elements
9. **K-way Merge**: Merging multiple sorted structures
10. **Dynamic Programming**: Optimization problems with overlapping subproblems
11. **Backtracking**: Generating all combinations/permutations
12. **Graph Traversal**: Connected components, shortest paths, cycles

## Time & Space Complexity Quick Reference

- **Two Pointers**: O(n) time, O(1) space
- **Sliding Window**: O(n) time, O(k) space
- **Binary Search**: O(log n) time, O(1) space
- **Tree Traversal**: O(n) time, O(h) space (h = height)
- **Graph Traversal**: O(V + E) time, O(V) space
- **Dynamic Programming**: Often O(n²) time, O(n) or O(n²) space
- **Backtracking**: O(2ⁿ) time in worst case
- **Heap Operations**: O(log n) insertion/deletion

Remember: The key to mastering these patterns is practice. Start with easier problems and gradually work your way up to more complex ones!

details - https://medium.com/@eshwarsairam7/dsa-patterns-cheat-sheet-836179d7a6cf
cleaner way to understand patterns - https://github.com/KushalVijay/DSA-Patterns-Roadmap
understand patterns in depth - https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews
