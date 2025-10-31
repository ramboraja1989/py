# DSA Patterns Cheatsheet 🚀

A comprehensive guide to common Data Structures and Algorithms patterns with Python implementations.

## Important Notes
- **ListNode Definition**: All linked list problems assume the standard definition:
  ```python
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next
  ```
- **TreeNode Definition**: All tree problems assume the standard definition:
  ```python
  class TreeNode:
      def __init__(self, val=0, left=None, right=None):
          self.val = val
          self.left = left
          self.right = right
  ```
- **Python Heaps**: Python's `heapq` module only provides min heap. For max heap, negate values.

---

## Table of Contents

### Part I: Data Structures
- [Python Data Structures Reference](#python-data-structures-reference)
  - [Lists (Arrays)](#1-lists-arrays)
  - [Strings](#2-strings)
  - [Dictionaries (Hash Maps)](#3-dictionaries-hash-maps)
  - [Sets](#4-sets)
  - [Tuples](#5-tuples)
  - [Heaps](#6-heaps)
  - [Deque](#7-deque)
  - [Counter](#8-counter)
  - [defaultdict](#9-defaultdict)
  - [OrderedDict](#10-ordereddict)
  - [bisect Module](#11-bisect-module)
  - [itertools Module](#12-itertools-module)
  - [Linked Lists](#13-linked-lists)
  - [Binary Trees](#14-binary-trees)
  - [Binary Search Trees](#15-binary-search-trees)
  - [Graphs](#16-graphs)
  - [Stacks](#17-stacks)
  - [Queues](#18-queues)

### Part II: Algorithm Patterns
1. [Two Pointers](#1-two-pointers)
2. [Sliding Window](#2-sliding-window)
3. [Fast & Slow Pointers](#3-fast--slow-pointers)
4. [Binary Search](#4-binary-search)
5. [Merge Intervals](#5-merge-intervals)
6. [In-place Reversal of Linked List](#6-in-place-reversal-of-linked-list)
7. [Tree BFS](#7-tree-bfs)
8. [Tree DFS](#8-tree-dfs)
9. [Two Heaps](#9-two-heaps)
10. [Subsets/Backtracking](#10-subsetsbacktracking)
11. [Top K Elements](#11-top-k-elements)
12. [Dynamic Programming](#12-dynamic-programming)
13. [Monotonic Stack](#13-monotonic-stack)
14. [Prefix Sum](#14-prefix-sum)
15. [Union Find](#15-union-find)
16. [Topological Sort](#16-topological-sort)
17. [Trie](#17-trie)

---

# Part I: Python Data Structures Reference

## 1. Lists (Arrays)

### Overview
Dynamic arrays that can store elements of any type. Most commonly used data structure in Python.

### Time Complexity
- **Access**: O(1)
- **Search**: O(n)
- **Insert at end**: O(1) amortized
- **Insert at position**: O(n)
- **Delete**: O(n)
- **Pop from end**: O(1)

### When to Use
- Need ordered collection with index access
- Frequent appending/popping from end
- Need to sort elements
- Building result arrays

### Initialization
```python
# Empty list
arr = []
arr = list()

# With size (default value None)
arr = [None] * 5  # [None, None, None, None, None]

# With default value
arr = [0] * 5  # [0, 0, 0, 0, 0]
arr = [False] * 5  # [False, False, False, False, False]

# From range
arr = list(range(5))  # [0, 1, 2, 3, 4]
arr = list(range(1, 6))  # [1, 2, 3, 4, 5]
arr = list(range(0, 10, 2))  # [0, 2, 4, 6, 8]

# 2D array
matrix = [[0] * cols for _ in range(rows)]  # CORRECT way
# DON'T: [[0] * cols] * rows  # This creates references to same list!

# List comprehension
arr = [i**2 for i in range(5)]  # [0, 1, 4, 9, 16]
arr = [i for i in range(10) if i % 2 == 0]  # [0, 2, 4, 6, 8]

# From string
arr = list("hello")  # ['h', 'e', 'l', 'l', 'o']
```

### Common Operations
```python
arr = [1, 2, 3, 4, 5]

# Append/Extend
arr.append(6)  # [1, 2, 3, 4, 5, 6]
arr.extend([7, 8])  # [1, 2, 3, 4, 5, 6, 7, 8]
arr += [9, 10]  # Same as extend

# Insert
arr.insert(0, 0)  # Insert at index 0

# Remove
arr.remove(3)  # Remove first occurrence of 3
val = arr.pop()  # Remove and return last element
val = arr.pop(0)  # Remove and return element at index 0

# Access
first = arr[0]
last = arr[-1]
sublist = arr[1:4]  # Elements at index 1, 2, 3
sublist = arr[:3]  # First 3 elements
sublist = arr[3:]  # From index 3 to end
sublist = arr[::2]  # Every 2nd element
reversed_arr = arr[::-1]  # Reverse the list

# Search
if 5 in arr:  # O(n) check
    index = arr.index(5)  # Get first index of 5
    
count = arr.count(5)  # Count occurrences

# Sort
arr.sort()  # In-place sort (ascending)
arr.sort(reverse=True)  # Descending
arr.sort(key=lambda x: abs(x))  # Custom sort
sorted_arr = sorted(arr)  # Return new sorted list

# Reverse
arr.reverse()  # In-place
reversed_arr = list(reversed(arr))  # Return new reversed list

# Min/Max/Sum
minimum = min(arr)
maximum = max(arr)
total = sum(arr)

# Length
length = len(arr)

# Clear
arr.clear()  # Remove all elements

# Copy
arr_copy = arr.copy()  # Shallow copy
arr_copy = arr[:]  # Also shallow copy
import copy
arr_copy = copy.deepcopy(arr)  # Deep copy for nested structures
```

### Useful Patterns
```python
# Enumerate (get index and value)
for i, val in enumerate(arr):
    print(f"Index {i}: {val}")

# Zip (iterate multiple lists together)
arr1 = [1, 2, 3]
arr2 = ['a', 'b', 'c']
for num, char in zip(arr1, arr2):
    print(num, char)

# Filter
filtered = [x for x in arr if x > 5]
filtered = list(filter(lambda x: x > 5, arr))

# Map
doubled = [x * 2 for x in arr]
doubled = list(map(lambda x: x * 2, arr))

# Any/All
has_even = any(x % 2 == 0 for x in arr)
all_positive = all(x > 0 for x in arr)

# Finding index of max/min
max_idx = arr.index(max(arr))
min_idx = arr.index(min(arr))
```

---

## 2. Strings

### Overview
Immutable sequence of characters. Cannot be modified in place.

### Time Complexity
- **Access**: O(1)
- **Search**: O(n)
- **Concatenation**: O(n+m)
- **Slicing**: O(k) where k is slice size

### When to Use
- Text processing
- Pattern matching
- Palindrome problems
- Character manipulation

### Initialization
```python
# Basic
s = "hello"
s = 'hello'
s = str(123)  # "123"

# Empty string
s = ""
s = str()

# From list
s = ''.join(['h', 'e', 'l', 'l', 'o'])  # "hello"
s = ','.join(['a', 'b', 'c'])  # "a,b,c"

# Multiline
s = """This is
a multiline
string"""

# Repeat
s = "ab" * 3  # "ababab"

# From characters
s = chr(97)  # 'a'
```

### Common Operations
```python
s = "Hello World"

# Access
first = s[0]  # 'H'
last = s[-1]  # 'd'
substr = s[0:5]  # "Hello"
substr = s[:5]  # "Hello"
substr = s[6:]  # "World"

# Length
length = len(s)  # 11

# Search
if "World" in s:  # True
    index = s.find("World")  # 6 (or -1 if not found)
    index = s.index("World")  # 6 (raises ValueError if not found)

count = s.count('l')  # 3

# Case
lower = s.lower()  # "hello world"
upper = s.upper()  # "HELLO WORLD"
title = s.title()  # "Hello World"
swapped = s.swapcase()  # "hELLO wORLD"

# Check
s.isalpha()  # Check if all alphabetic
s.isdigit()  # Check if all digits
s.isalnum()  # Check if alphanumeric
s.isspace()  # Check if all whitespace
s.islower()  # Check if all lowercase
s.isupper()  # Check if all uppercase
s.startswith("Hello")  # True
s.endswith("World")  # True

# Split/Join
words = s.split()  # ['Hello', 'World']
words = s.split(',')  # Split by comma
result = ' '.join(words)  # Join with space

# Strip (remove whitespace)
s = "  hello  "
trimmed = s.strip()  # "hello"
trimmed = s.lstrip()  # "hello  "
trimmed = s.rstrip()  # "  hello"

# Replace (returns new string)
new_s = s.replace("Hello", "Hi")  # "Hi World"
new_s = s.replace("l", "L", 2)  # Replace first 2 occurrences

# Convert to list (for modification)
chars = list(s)
chars[0] = 'h'
s = ''.join(chars)

# Reverse
reversed_s = s[::-1]

# Character conversions
ord('A')  # 65 (ASCII value)
chr(65)  # 'A'
```

### Useful Patterns
```python
# Check palindrome
def is_palindrome(s):
    return s == s[::-1]

# Remove non-alphanumeric
cleaned = ''.join(c for c in s if c.isalnum())

# Count characters
from collections import Counter
char_count = Counter(s)

# String formatting
name = "Alice"
age = 25
formatted = f"Name: {name}, Age: {age}"  # f-string (Python 3.6+)
formatted = "Name: {}, Age: {}".format(name, age)
formatted = "Name: %s, Age: %d" % (name, age)

# Padding
s = "hello"
padded = s.center(10)  # "  hello   "
padded = s.ljust(10, '*')  # "hello*****"
padded = s.rjust(10, '*')  # "*****hello"
padded = s.zfill(10)  # "00000hello" (for numbers)
```

---

## 3. Dictionaries (Hash Maps)

### Overview
Key-value pairs with O(1) average-case lookup. Keys must be immutable.

### Time Complexity
- **Access**: O(1) average, O(n) worst
- **Insert**: O(1) average
- **Delete**: O(1) average
- **Search**: O(1) average

### When to Use
- Need fast lookups by key
- Counting occurrences
- Memoization
- Graph adjacency lists
- Two-sum type problems

### Initialization
```python
# Empty dict
d = {}
d = dict()

# With initial values
d = {'a': 1, 'b': 2, 'c': 3}
d = dict(a=1, b=2, c=3)
d = dict([('a', 1), ('b', 2)])

# From two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
d = dict(zip(keys, values))

# Dict comprehension
d = {i: i**2 for i in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Default value for all keys
d = dict.fromkeys(['a', 'b', 'c'], 0)  # {'a': 0, 'b': 0, 'c': 0}
```

### Common Operations
```python
d = {'a': 1, 'b': 2, 'c': 3}

# Access
val = d['a']  # 1 (KeyError if key doesn't exist)
val = d.get('a')  # 1
val = d.get('d', 0)  # 0 (default if key doesn't exist)

# Insert/Update
d['d'] = 4
d.update({'e': 5, 'f': 6})
d.update(g=7, h=8)

# Delete
del d['a']
val = d.pop('b')  # Remove and return value
val = d.pop('z', 0)  # Return default if key doesn't exist
d.popitem()  # Remove and return last (key, value) pair
d.clear()  # Remove all items

# Check existence
if 'a' in d:  # Check key exists
    pass
if 'a' not in d:
    pass

# Get keys/values/items
keys = d.keys()  # dict_keys object
values = d.values()  # dict_values object
items = d.items()  # dict_items object

keys_list = list(d.keys())
values_list = list(d.values())

# Iteration
for key in d:
    print(key, d[key])

for key, value in d.items():
    print(key, value)

# Copy
d_copy = d.copy()  # Shallow copy
import copy
d_copy = copy.deepcopy(d)  # Deep copy

# Merge dictionaries (Python 3.9+)
d1 = {'a': 1, 'b': 2}
d2 = {'c': 3, 'd': 4}
merged = d1 | d2  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
d1.update(d2)  # In-place merge

# Length
length = len(d)
```

### Useful Patterns
```python
# Count occurrences
arr = [1, 2, 2, 3, 3, 3]
count = {}
for num in arr:
    count[num] = count.get(num, 0) + 1

# Or use Counter (better)
from collections import Counter
count = Counter(arr)

# Group by key
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[item.category].append(item)

# Get key with max value
max_key = max(d, key=d.get)

# Sort by key
sorted_dict = dict(sorted(d.items()))

# Sort by value
sorted_dict = dict(sorted(d.items(), key=lambda x: x[1]))

# Reverse key-value
reversed_dict = {v: k for k, v in d.items()}

# Filter dict
filtered = {k: v for k, v in d.items() if v > 5}

# Two Sum pattern
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

---

## 4. Sets

### Overview
Unordered collection of unique elements. Built on hash table.

### Time Complexity
- **Add**: O(1) average
- **Remove**: O(1) average
- **Search**: O(1) average
- **Union/Intersection**: O(min(len(s1), len(s2)))

### When to Use
- Remove duplicates
- Membership testing
- Finding unique elements
- Set operations (union, intersection)

### Initialization
```python
# Empty set
s = set()  # Note: {} creates empty dict, not set!

# With initial values
s = {1, 2, 3, 4, 5}
s = set([1, 2, 3, 4, 5])
s = set("hello")  # {'h', 'e', 'l', 'o'}

# Set comprehension
s = {i**2 for i in range(5)}  # {0, 1, 4, 9, 16}

# From list (removes duplicates)
arr = [1, 2, 2, 3, 3, 3]
s = set(arr)  # {1, 2, 3}
```

### Common Operations
```python
s = {1, 2, 3, 4, 5}

# Add/Remove
s.add(6)
s.remove(3)  # KeyError if not exists
s.discard(3)  # No error if not exists
elem = s.pop()  # Remove and return arbitrary element
s.clear()

# Check existence
if 5 in s:  # O(1)
    pass

# Set operations
s1 = {1, 2, 3}
s2 = {3, 4, 5}

union = s1 | s2  # {1, 2, 3, 4, 5}
union = s1.union(s2)

intersection = s1 & s2  # {3}
intersection = s1.intersection(s2)

difference = s1 - s2  # {1, 2}
difference = s1.difference(s2)

symmetric_diff = s1 ^ s2  # {1, 2, 4, 5}
symmetric_diff = s1.symmetric_difference(s2)

# Subset/Superset
is_subset = s1 <= s2
is_subset = s1.issubset(s2)

is_superset = s1 >= s2
is_superset = s1.issuperset(s2)

is_disjoint = s1.isdisjoint(s2)  # No common elements

# Update operations (in-place)
s1 |= s2  # Union
s1.update(s2)

s1 &= s2  # Intersection
s1.intersection_update(s2)

s1 -= s2  # Difference
s1.difference_update(s2)

# Length
length = len(s)

# Iteration
for elem in s:
    print(elem)

# Convert to list
arr = list(s)
```

### Useful Patterns
```python
# Remove duplicates from list preserving order
def remove_duplicates(arr):
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Find unique elements
arr1 = [1, 2, 3, 4]
arr2 = [3, 4, 5, 6]
unique_in_arr1 = list(set(arr1) - set(arr2))  # [1, 2]

# Check if all elements unique
def all_unique(arr):
    return len(arr) == len(set(arr))

# Find common elements
common = set(arr1) & set(arr2)

# Frozen set (immutable set, can be dict key)
fs = frozenset([1, 2, 3])
d = {fs: "value"}
```

---

## 5. Tuples

### Overview
Immutable ordered sequences. Can be used as dictionary keys.

### Time Complexity
- **Access**: O(1)
- **Search**: O(n)

### When to Use
- Immutable data
- Return multiple values from function
- Dictionary keys (when you need composite keys)
- Coordinates, points

### Initialization
```python
# Basic
t = (1, 2, 3)
t = 1, 2, 3  # Parentheses optional
t = tuple([1, 2, 3])

# Single element tuple
t = (1,)  # Comma is required!
t = 1,

# Empty tuple
t = ()
t = tuple()

# From string
t = tuple("hello")  # ('h', 'e', 'l', 'l', 'o')
```

### Common Operations
```python
t = (1, 2, 3, 4, 5)

# Access
first = t[0]
last = t[-1]
subtuple = t[1:4]

# Search
if 3 in t:
    index = t.index(3)
count = t.count(3)

# Length
length = len(t)

# Unpacking
a, b, c = (1, 2, 3)
first, *rest = (1, 2, 3, 4)  # first=1, rest=[2,3,4]
first, *middle, last = (1, 2, 3, 4, 5)  # first=1, middle=[2,3,4], last=5

# Concatenation
t1 = (1, 2)
t2 = (3, 4)
combined = t1 + t2  # (1, 2, 3, 4)

# Repeat
repeated = (1, 2) * 3  # (1, 2, 1, 2, 1, 2)

# Iteration
for item in t:
    print(item)

# Convert to list
arr = list(t)
```

### Useful Patterns
```python
# Return multiple values
def min_max(arr):
    return min(arr), max(arr)

minimum, maximum = min_max([1, 2, 3, 4, 5])

# Use as dictionary key
coordinates = {(0, 0): "origin", (1, 1): "diagonal"}

# Swap variables
a, b = b, a

# Named tuples (more readable)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)  # 1 2
```

---

## 6. Heaps

### Overview
Binary heap implementation (min heap only). Useful for priority queue.

### Time Complexity
- **Push**: O(log n)
- **Pop**: O(log n)
- **Peek**: O(1)
- **Heapify**: O(n)

### When to Use
- Find K largest/smallest elements
- Priority queue
- Median finding
- Merge K sorted lists

### Initialization & Operations
```python
import heapq

# Create empty heap
heap = []

# Create heap from list
arr = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(arr)  # Convert to heap in-place, O(n)

# Push element
heapq.heappush(heap, 5)

# Pop smallest element
smallest = heapq.heappop(heap)

# Push and pop in one operation
heapq.heappushpop(heap, 5)  # More efficient than push then pop

# Replace smallest (pop then push)
heapq.heapreplace(heap, 5)

# Peek at smallest (don't remove)
smallest = heap[0]  # Min element always at index 0

# Get n smallest/largest
arr = [3, 1, 4, 1, 5, 9, 2, 6]
n_smallest = heapq.nsmallest(3, arr)  # [1, 1, 2]
n_largest = heapq.nlargest(3, arr)  # [9, 6, 5]

# With custom key
data = [('A', 3), ('B', 1), ('C', 2)]
smallest = heapq.nsmallest(2, data, key=lambda x: x[1])  # [('B', 1), ('C', 2)]
```

### Max Heap (Using Negation)
```python
import heapq

# Max heap by negating values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)

# Pop largest (most negative)
largest = -heapq.heappop(max_heap)  # 7

# Peek at largest
largest = -max_heap[0]  # 5
```

### Useful Patterns
```python
# K largest elements
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# K smallest elements using heap of size k
def k_smallest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, -num)  # Max heap
        if len(heap) > k:
            heapq.heappop(heap)
    return [-x for x in heap]

# Merge K sorted lists
def merge_k_sorted(lists):
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_idx, element_idx)
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

# Find running median
class MedianFinder:
    def __init__(self):
        self.max_heap = []  # Lower half
        self.min_heap = []  # Upper half
    
    def addNum(self, num):
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

---

## 7. Deque

### Overview
Double-ended queue. Efficient O(1) operations at both ends.

### Time Complexity
- **Append/Pop (both ends)**: O(1)
- **Access by index**: O(n)
- **Insert/Remove (middle)**: O(n)

### When to Use
- Sliding window problems
- BFS in graphs/trees
- Maintaining a window of elements
- Implementing stack or queue

### Initialization & Operations
```python
from collections import deque

# Create deque
dq = deque()
dq = deque([1, 2, 3, 4, 5])
dq = deque([1, 2, 3], maxlen=3)  # Fixed size, auto-removes from other end

# Append
dq.append(6)  # Add to right: [1, 2, 3, 4, 5, 6]
dq.appendleft(0)  # Add to left: [0, 1, 2, 3, 4, 5, 6]

# Extend
dq.extend([7, 8])  # Add multiple to right
dq.extendleft([- 1, -2])  # Add multiple to left (reversed!)

# Pop
right = dq.pop()  # Remove from right
left = dq.popleft()  # Remove from left

# Access
first = dq[0]
last = dq[-1]

# Rotate
dq = deque([1, 2, 3, 4, 5])
dq.rotate(2)  # [4, 5, 1, 2, 3] (rotate right)
dq.rotate(-2)  # [1, 2, 3, 4, 5] (rotate left)

# Other operations
dq.remove(3)  # Remove first occurrence
count = dq.count(2)  # Count occurrences
dq.reverse()  # Reverse in-place
dq.clear()  # Remove all elements

# Length
length = len(dq)

# Check if empty
if dq:  # Non-empty
    pass
```

### Useful Patterns
```python
# BFS traversal
from collections import deque

def bfs(root):
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

# Sliding window maximum
def maxSlidingWindow(nums, k):
    from collections import deque
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they won't be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Implement stack using deque
stack = deque()
stack.append(1)  # Push
top = stack.pop()  # Pop

# Implement queue using deque
queue = deque()
queue.append(1)  # Enqueue
front = queue.popleft()  # Dequeue
```

---

## 8. Counter

### Overview
Dictionary subclass for counting hashable objects. Specialized for counting.

### Time Complexity
- **Count**: O(n) to create
- **Access**: O(1)
- **Most common**: O(n log k) for k elements

### When to Use
- Counting frequencies
- Finding most/least common elements
- Comparing element counts
- Multiset operations

### Initialization & Operations
```python
from collections import Counter

# Create counter
c = Counter()
c = Counter([1, 2, 2, 3, 3, 3])  # Counter({3: 3, 2: 2, 1: 1})
c = Counter("hello world")  # Counter({'l': 3, 'o': 2, ...})
c = Counter({'a': 4, 'b': 2})
c = Counter(a=4, b=2)

# Access counts
count = c['a']  # Returns 0 if key doesn't exist (not KeyError)
count = c.get('a', 0)

# Update counts
c = Counter(['a', 'b'])
c.update(['a', 'c'])  # Counter({'a': 2, 'b': 1, 'c': 1})
c['a'] += 1

# Most/Least common
c = Counter(['a', 'a', 'a', 'b', 'b', 'c'])
most_common = c.most_common(2)  # [('a', 3), ('b', 2)]
least_common = c.most_common()[:-3:-1]  # Get least common

# Elements (returns iterator)
c = Counter(a=3, b=2)
elements = list(c.elements())  # ['a', 'a', 'a', 'b', 'b']

# Arithmetic operations
c1 = Counter(['a', 'b', 'b'])
c2 = Counter(['b', 'c'])

addition = c1 + c2  # Counter({'b': 3, 'a': 1, 'c': 1})
subtraction = c1 - c2  # Counter({'a': 1, 'b': 1})
intersection = c1 & c2  # Counter({'b': 1}) - min(c1[x], c2[x])
union = c1 | c2  # Counter({'b': 2, 'a': 1, 'c': 1}) - max(c1[x], c2[x])

# Delete zeros and negatives
c = Counter(a=2, b=-1, c=0)
+c  # Counter({'a': 2}) - removes non-positive counts
```

### Useful Patterns
```python
# Find most frequent element
def most_frequent(arr):
    c = Counter(arr)
    return c.most_common(1)[0][0]

# Check if two strings are anagrams
def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

# Find first non-repeating character
def first_unique_char(s):
    c = Counter(s)
    for i, char in enumerate(s):
        if c[char] == 1:
            return i
    return -1

# Top K frequent elements
def top_k_frequent(nums, k):
    c = Counter(nums)
    return [num for num, _ in c.most_common(k)]

# Check if array can be divided into pairs
def can_pair(arr):
    c = Counter(arr)
    return all(count % 2 == 0 for count in c.values())
```

---

## 9. defaultdict

### Overview
Dictionary subclass that provides default values for missing keys.

### Time Complexity
Same as regular dict - O(1) average for all operations

### When to Use
- Avoid key existence checks
- Group items by category
- Build graphs (adjacency lists)
- Count/accumulate values

### Initialization & Operations
```python
from collections import defaultdict

# Create with default int (0)
d = defaultdict(int)
d['a'] += 1  # No KeyError, starts from 0

# Create with default list
d = defaultdict(list)
d['key'].append(value)  # No need to check if key exists

# Create with default set
d = defaultdict(set)
d['key'].add(value)

# Create with default dict
d = defaultdict(dict)

# Create with custom default
d = defaultdict(lambda: "default_value")

# Create with default factory
def default_factory():
    return []
d = defaultdict(default_factory)

# Convert to regular dict
regular_dict = dict(d)
```

### Useful Patterns
```python
# Group anagrams
def group_anagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# Build graph adjacency list
def build_graph(edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # For undirected graph
    return graph

# Count word frequencies
def word_frequency(text):
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1
    return word_count

# Group by property
def group_by_length(words):
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return groups

# Tree structure using defaultdict
def make_tree():
    return defaultdict(make_tree)

tree = make_tree()
tree['animals']['mammals']['dog'] = 'bark'
tree['animals']['mammals']['cat'] = 'meow'
```

---

## 10. OrderedDict

### Overview
Dictionary that remembers insertion order. (Note: Regular dicts preserve order in Python 3.7+)

### When to Use
- Need to remember insertion order (Python < 3.7)
- Need to reorder items
- LRU Cache implementation
- Move items to end/beginning

### Initialization & Operations
```python
from collections import OrderedDict

# Create
od = OrderedDict()
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])

# Move to end
od.move_to_end('a')  # Move 'a' to end
od.move_to_end('c', last=False)  # Move 'c' to beginning

# Pop items
last_item = od.popitem()  # Remove and return last (key, value)
first_item = od.popitem(last=False)  # Remove and return first

# All regular dict operations work
od['d'] = 4
del od['b']
```

### Useful Patterns
```python
# LRU Cache implementation
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used

# Maintain insertion order with updates
od = OrderedDict([('a', 1), ('b', 2)])
od['a'] = 10  # Update value, order unchanged
```

---

## 11. bisect Module

### Overview
Binary search and insertion functions for sorted lists.

### Time Complexity
- **bisect/insort**: O(log n) for search, O(n) for insert

### When to Use
- Binary search in sorted list
- Insert while maintaining sorted order
- Find insertion point for value

### Operations
```python
import bisect

arr = [1, 3, 5, 7, 9]

# Find insertion point (if duplicates exist, insert after)
pos = bisect.bisect_right(arr, 5)  # 3 (after 5)
pos = bisect.bisect(arr, 5)  # Same as bisect_right

# Find insertion point (if duplicates exist, insert before)
pos = bisect.bisect_left(arr, 5)  # 2 (before 5)

# Insert and keep sorted
bisect.insort_right(arr, 4)  # arr = [1, 3, 4, 5, 7, 9]
bisect.insort(arr, 4)  # Same as insort_right

bisect.insort_left(arr, 4)

# Find range of target
def find_range(arr, target):
    left = bisect.bisect_left(arr, target)
    right = bisect.bisect_right(arr, target)
    if left < right:
        return [left, right - 1]
    return [-1, -1]
```

### Useful Patterns
```python
# Binary search (check if exists)
def binary_search(arr, target):
    i = bisect.bisect_left(arr, target)
    return i < len(arr) and arr[i] == target

# Find first element >= target
def lower_bound(arr, target):
    return bisect.bisect_left(arr, target)

# Find first element > target
def upper_bound(arr, target):
    return bisect.bisect_right(arr, target)

# Count elements in range [left, right]
def count_range(arr, left, right):
    return bisect.bisect_right(arr, right) - bisect.bisect_left(arr, left)

# Maintain sorted list with duplicates
class SortedList:
    def __init__(self):
        self.items = []
    
    def add(self, val):
        bisect.insort(self.items, val)
    
    def remove(self, val):
        i = bisect.bisect_left(self.items, val)
        if i < len(self.items) and self.items[i] == val:
            self.items.pop(i)
```

---

## 12. itertools Module

### Overview
Functions for efficient looping and combinatorial operations.

### Common Functions
```python
import itertools

# Infinite iterators
# count(start, step)
for i in itertools.count(10, 2):  # 10, 12, 14, 16, ...
    if i > 20:
        break

# cycle(iterable) - repeat infinitely
for item in itertools.cycle(['A', 'B', 'C']):  # A, B, C, A, B, C, ...
    break  # Would go forever!

# repeat(value, times)
list(itertools.repeat(5, 3))  # [5, 5, 5]

# Combinatorial iterators
# product - Cartesian product
list(itertools.product([1, 2], ['a', 'b']))  
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

list(itertools.product([1, 2], repeat=2))  
# [(1, 1), (1, 2), (2, 1), (2, 2)]

# permutations - All orderings
list(itertools.permutations([1, 2, 3], 2))  
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# combinations - Choose r from n (no repeats, order doesn't matter)
list(itertools.combinations([1, 2, 3, 4], 2))  
# [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# combinations_with_replacement
list(itertools.combinations_with_replacement([1, 2, 3], 2))  
# [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

# Terminating iterators
# chain - Flatten multiple iterables
list(itertools.chain([1, 2], [3, 4], [5]))  # [1, 2, 3, 4, 5]

# compress - Filter by boolean mask
list(itertools.compress([1, 2, 3, 4], [1, 0, 1, 0]))  # [1, 3]

# dropwhile - Drop elements while predicate is true
list(itertools.dropwhile(lambda x: x < 3, [1, 2, 3, 4, 1]))  # [3, 4, 1]

# takewhile - Take elements while predicate is true
list(itertools.takewhile(lambda x: x < 3, [1, 2, 3, 4]))  # [1, 2]

# groupby - Group consecutive elements
data = [1, 1, 2, 2, 2, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(key, list(group))
# 1 [1, 1]
# 2 [2, 2, 2]
# 3 [3]
# 1 [1, 1]

# islice - Slice iterator
list(itertools.islice(range(10), 2, 8, 2))  # [2, 4, 6]

# zip_longest - Zip with fillvalue for unequal lengths
list(itertools.zip_longest([1, 2], ['a', 'b', 'c'], fillvalue=0))
# [(1, 'a'), (2, 'b'), (0, 'c')]

# accumulate - Running totals
list(itertools.accumulate([1, 2, 3, 4]))  # [1, 3, 6, 10]
list(itertools.accumulate([1, 2, 3, 4], lambda x, y: x * y))  # [1, 2, 6, 24]

# pairwise (Python 3.10+) - Consecutive pairs
# list(itertools.pairwise([1, 2, 3, 4]))  # [(1, 2), (2, 3), (3, 4)]
```

### Useful Patterns
```python
# All possible pairs
def all_pairs(arr):
    return list(itertools.combinations(arr, 2))

# Flatten nested list
nested = [[1, 2], [3, 4], [5]]
flat = list(itertools.chain.from_iterable(nested))  # [1, 2, 3, 4, 5]

# Sliding window
def sliding_window(iterable, n):
    it = iter(iterable)
    window = tuple(itertools.islice(it, n))
    if len(window) == n:
        yield window
    for elem in it:
        window = window[1:] + (elem,)
        yield window

# Generate all binary strings of length n
def binary_strings(n):
    return [''.join(p) for p in itertools.product('01', repeat=n)]

# Chunk iterator into groups of n
def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk
```

---

## 13. Linked Lists

### Overview
Linear data structure where elements are stored in nodes, each pointing to the next node.

### Node Definition
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Time Complexity
- **Access**: O(n)
- **Search**: O(n)
- **Insert at head**: O(1)
- **Insert at tail**: O(n) without tail pointer, O(1) with tail pointer
- **Insert at position**: O(n)
- **Delete**: O(n)

### When to Use
- Frequent insertions/deletions at beginning
- Unknown size or dynamic size
- No random access needed
- Implementing stacks, queues, or LRU cache

### Basic Operations
```python
class LinkedList:
    def __init__(self):
        self.head = None
    
    # Insert at beginning - O(1)
    def insert_at_head(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    # Insert at end - O(n)
    def insert_at_tail(self, val):
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    # Insert at position - O(n)
    def insert_at_position(self, pos, val):
        if pos == 0:
            self.insert_at_head(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        
        for _ in range(pos - 1):
            if not current:
                return
            current = current.next
        
        if current:
            new_node.next = current.next
            current.next = new_node
    
    # Delete node with value - O(n)
    def delete(self, val):
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next
    
    # Search - O(n)
    def search(self, val):
        current = self.head
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    
    # Get length - O(n)
    def length(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count
    
    # Reverse - O(n)
    def reverse(self):
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    # Display - O(n)
    def display(self):
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
```

### Common Patterns
```python
# Create linked list from array
def create_linked_list(arr):
    if not arr:
        return None
    
    head = ListNode(arr[0])
    current = head
    
    for i in range(1, len(arr)):
        current.next = ListNode(arr[i])
        current = current.next
    
    return head

# Convert linked list to array
def linked_list_to_array(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

# Find middle (Fast & Slow pointers)
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# Detect cycle
def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False

# Merge two sorted lists
def merge_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 if l1 else l2
    return dummy.next

# Remove nth node from end
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    
    # Move first n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove nth node
    second.next = second.next.next
    return dummy.next
```

### Doubly Linked List
```python
class DListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def insert_at_head(self, val):
        new_node = DListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
    
    def insert_at_tail(self, val):
        new_node = DListNode(val)
        
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
    
    def delete_node(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
```

---

## 14. Binary Trees

### Overview
Hierarchical data structure where each node has at most two children.

### Node Definition
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Time Complexity
- **Access**: O(n)
- **Search**: O(n)
- **Insert**: O(n)
- **Delete**: O(n)
- **Traversal**: O(n)

### When to Use
- Hierarchical data representation
- Expression evaluation
- File system structure
- Decision trees

### Traversal Methods
```python
# Inorder Traversal (Left, Root, Right) - Gives sorted order for BST
def inorder(root):
    result = []
    
    def helper(node):
        if not node:
            return
        helper(node.left)
        result.append(node.val)
        helper(node.right)
    
    helper(root)
    return result

# Iterative Inorder
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result

# Preorder Traversal (Root, Left, Right)
def preorder(root):
    result = []
    
    def helper(node):
        if not node:
            return
        result.append(node.val)
        helper(node.left)
        helper(node.right)
    
    helper(root)
    return result

# Iterative Preorder
def preorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# Postorder Traversal (Left, Right, Root)
def postorder(root):
    result = []
    
    def helper(node):
        if not node:
            return
        helper(node.left)
        helper(node.right)
        result.append(node.val)
    
    helper(root)
    return result

# Level Order Traversal (BFS)
def level_order(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Common Operations
```python
# Height/Depth of tree
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Count nodes
def count_nodes(root):
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)

# Check if balanced
def is_balanced(root):
    def helper(node):
        if not node:
            return 0, True
        
        left_height, left_balanced = helper(node.left)
        right_height, right_balanced = helper(node.right)
        
        balanced = (left_balanced and right_balanced and 
                   abs(left_height - right_height) <= 1)
        
        return 1 + max(left_height, right_height), balanced
    
    return helper(root)[1]

# Check if symmetric
def is_symmetric(root):
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    return is_mirror(root, root)

# Lowest Common Ancestor
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right

# Path sum - check if path from root to leaf sums to target
def has_path_sum(root, target_sum):
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == target_sum
    
    return (has_path_sum(root.left, target_sum - root.val) or
            has_path_sum(root.right, target_sum - root.val))

# All paths from root to leaf
def all_paths(root):
    if not root:
        return []
    
    result = []
    
    def dfs(node, path):
        if not node:
            return
        
        path.append(node.val)
        
        if not node.left and not node.right:
            result.append(path[:])
        else:
            dfs(node.left, path)
            dfs(node.right, path)
        
        path.pop()
    
    dfs(root, [])
    return result

# Serialize and Deserialize
def serialize(root):
    if not root:
        return "None"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    def helper(nodes):
        val = next(nodes)
        if val == "None":
            return None
        node = TreeNode(int(val))
        node.left = helper(nodes)
        node.right = helper(nodes)
        return node
    
    return helper(iter(data.split(',')))

# Build tree from inorder and preorder
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    mid = inorder.index(root_val)
    
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
    
    return root
```

---

## 15. Binary Search Trees

### Overview
Binary tree where left child < parent < right child. Enables O(log n) operations.

### Time Complexity (Balanced BST)
- **Search**: O(log n) average, O(n) worst
- **Insert**: O(log n) average, O(n) worst
- **Delete**: O(log n) average, O(n) worst
- **Min/Max**: O(log n) average, O(n) worst

### When to Use
- Need sorted data with fast search
- Range queries
- Finding predecessor/successor
- Implementing sets and maps

### Basic Operations
```python
class BST:
    def __init__(self):
        self.root = None
    
    # Insert - O(log n) average
    def insert(self, val):
        def helper(node, val):
            if not node:
                return TreeNode(val)
            
            if val < node.val:
                node.left = helper(node.left, val)
            elif val > node.val:
                node.right = helper(node.right, val)
            
            return node
        
        self.root = helper(self.root, val)
    
    # Search - O(log n) average
    def search(self, val):
        current = self.root
        
        while current:
            if val == current.val:
                return True
            elif val < current.val:
                current = current.left
            else:
                current = current.right
        
        return False
    
    # Find minimum - O(log n)
    def find_min(self):
        if not self.root:
            return None
        
        current = self.root
        while current.left:
            current = current.left
        
        return current.val
    
    # Find maximum - O(log n)
    def find_max(self):
        if not self.root:
            return None
        
        current = self.root
        while current.right:
            current = current.right
        
        return current.val
    
    # Delete - O(log n) average
    def delete(self, val):
        def helper(node, val):
            if not node:
                return None
            
            if val < node.val:
                node.left = helper(node.left, val)
            elif val > node.val:
                node.right = helper(node.right, val)
            else:
                # Node to delete found
                # Case 1: No children or one child
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                
                # Case 2: Two children
                # Find inorder successor (smallest in right subtree)
                min_node = node.right
                while min_node.left:
                    min_node = min_node.left
                
                node.val = min_node.val
                node.right = helper(node.right, min_node.val)
            
            return node
        
        self.root = helper(self.root, val)
```

### Common BST Problems
```python
# Validate BST
def is_valid_bst(root):
    def helper(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (helper(node.left, min_val, node.val) and
                helper(node.right, node.val, max_val))
    
    return helper(root, float('-inf'), float('inf'))

# Kth smallest element
def kth_smallest(root, k):
    stack = []
    current = root
    count = 0
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        count += 1
        
        if count == k:
            return current.val
        
        current = current.right

# Range sum BST
def range_sum_bst(root, low, high):
    if not root:
        return 0
    
    total = 0
    
    if low <= root.val <= high:
        total += root.val
    
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total

# Inorder successor
def inorder_successor(root, p):
    successor = None
    
    while root:
        if p.val < root.val:
            successor = root
            root = root.left
        else:
            root = root.right
    
    return successor

# Convert sorted array to BST
def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root
```

---

## 16. Graphs

### Overview
Collection of nodes (vertices) connected by edges. Can be directed or undirected.

### Representations
```python
# 1. Adjacency List (Most common for sparse graphs)
class Graph:
    def __init__(self):
        self.graph = {}  # or defaultdict(list)
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        # For undirected graph, add reverse edge
        # if v not in self.graph:
        #     self.graph[v] = []
        # self.graph[v].append(u)

# Using defaultdict
from collections import defaultdict

graph = defaultdict(list)
graph[0].append(1)
graph[0].append(2)

# 2. Adjacency Matrix (Dense graphs)
n = 5  # Number of vertices
adj_matrix = [[0] * n for _ in range(n)]
adj_matrix[0][1] = 1  # Edge from 0 to 1
adj_matrix[1][0] = 1  # Edge from 1 to 0 (undirected)

# 3. Edge List
edges = [(0, 1), (1, 2), (2, 3)]
```

### Time Complexity
- **Add vertex**: O(1)
- **Add edge**: O(1)
- **Remove vertex**: O(V + E)
- **Remove edge**: O(E)
- **Query edge**: O(1) matrix, O(degree) list

### Graph Traversals
```python
# DFS - Recursive
def dfs_recursive(graph, start):
    visited = set()
    result = []
    
    def dfs(node):
        visited.add(node)
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)
    return result

# DFS - Iterative
def dfs_iterative(graph, start):
    visited = set()
    result = []
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Add neighbors in reverse order for same order as recursive
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

# BFS
def bfs(graph, start):
    from collections import deque
    
    visited = set([start])
    result = []
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

### Common Graph Algorithms
```python
# Check if path exists
def has_path(graph, start, end):
    if start == end:
        return True
    
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node == end:
            return True
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                stack.append(neighbor)
    
    return False

# Shortest path (BFS for unweighted graph)
def shortest_path(graph, start, end):
    from collections import deque
    
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []

# Detect cycle in undirected graph (DFS)
def has_cycle_undirected(graph, n):
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True
    
    return False

# Detect cycle in directed graph (DFS with colors)
def has_cycle_directed(graph, n):
    # 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
    state = [0] * n
    
    def dfs(node):
        if state[node] == 1:  # Back edge - cycle found
            return True
        if state[node] == 2:  # Already processed
            return False
        
        state[node] = 1  # Mark as visiting
        
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        
        state[node] = 2  # Mark as visited
        return False
    
    for i in range(n):
        if state[i] == 0:
            if dfs(i):
                return True
    
    return False

# Topological Sort (Kahn's algorithm)
def topological_sort(n, edges):
    from collections import deque
    
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []

# Number of connected components (Union Find)
def count_components(n, edges):
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y
            return True
        return False
    
    components = n
    for u, v in edges:
        if union(u, v):
            components -= 1
    
    return components

# Dijkstra's Algorithm (Shortest path with weights)
def dijkstra(graph, start, n):
    import heapq
    
    distances = [float('inf')] * n
    distances[start] = 0
    heap = [(0, start)]  # (distance, node)
    
    while heap:
        dist, node = heapq.heappop(heap)
        
        if dist > distances[node]:
            continue
        
        for neighbor, weight in graph.get(node, []):
            new_dist = dist + weight
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return distances
```

### Graph as Grid (Common in LeetCode)
```python
# DFS on grid
def dfs_grid(grid, i, j):
    if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or 
        grid[i][j] == 0):
        return
    
    grid[i][j] = 0  # Mark as visited
    
    # Visit all 4 directions
    dfs_grid(grid, i+1, j)
    dfs_grid(grid, i-1, j)
    dfs_grid(grid, i, j+1)
    dfs_grid(grid, i, j-1)

# BFS on grid
def bfs_grid(grid, start_i, start_j):
    from collections import deque
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([(start_i, start_j)])
    visited.add((start_i, start_j))
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        i, j = queue.popleft()
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            if (0 <= ni < rows and 0 <= nj < cols and 
                (ni, nj) not in visited and grid[ni][nj] == 1):
                visited.add((ni, nj))
                queue.append((ni, nj))
```

---

## 17. Stacks

### Overview
LIFO (Last In First Out) data structure. Can use Python list.

### Time Complexity
- **Push**: O(1)
- **Pop**: O(1)
- **Peek**: O(1)
- **Search**: O(n)

### When to Use
- Function call stack
- Undo mechanisms
- Expression evaluation
- Backtracking
- Balanced parentheses

### Implementation
```python
# Using Python list (easiest)
stack = []
stack.append(1)  # Push
top = stack.pop()  # Pop
top = stack[-1] if stack else None  # Peek
is_empty = len(stack) == 0

# Custom implementation
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Using deque (also efficient)
from collections import deque
stack = deque()
stack.append(1)
top = stack.pop()
```

### Common Patterns
```python
# Valid parentheses
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    
    return not stack

# Evaluate postfix expression
def eval_postfix(expression):
    stack = []
    
    for token in expression.split():
        if token in ['+', '-', '*', '/']:
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]

# Next greater element (Monotonic stack)
def next_greater_elements(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

# Min stack (with O(1) min operation)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val
    
    def top(self):
        return self.stack[-1] if self.stack else None
    
    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None
```

---

## 18. Queues

### Overview
FIFO (First In First Out) data structure.

### Time Complexity
- **Enqueue**: O(1)
- **Dequeue**: O(1)
- **Front/Peek**: O(1)
- **Search**: O(n)

### When to Use
- BFS traversal
- Level order traversal
- Task scheduling
- Buffer/cache implementation
- Printer queue

### Implementation
```python
# Using deque (recommended)
from collections import deque

queue = deque()
queue.append(1)  # Enqueue
front = queue.popleft()  # Dequeue
front = queue[0] if queue else None  # Peek

# Using list (NOT recommended - O(n) for dequeue)
queue = []
queue.append(1)  # Enqueue
front = queue.pop(0)  # Dequeue - O(n)!

# Custom implementation using list
class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("dequeue from empty queue")
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("front from empty queue")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Circular Queue
class CircularQueue:
    def __init__(self, k):
        self.queue = [None] * k
        self.max_size = k
        self.head = 0
        self.tail = -1
        self.size = 0
    
    def enqueue(self, value):
        if self.is_full():
            return False
        
        self.tail = (self.tail + 1) % self.max_size
        self.queue[self.tail] = value
        self.size += 1
        return True
    
    def dequeue(self):
        if self.is_empty():
            return False
        
        self.head = (self.head + 1) % self.max_size
        self.size -= 1
        return True
    
    def front(self):
        return -1 if self.is_empty() else self.queue[self.head]
    
    def rear(self):
        return -1 if self.is_empty() else self.queue[self.tail]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.max_size
```

### Common Patterns
```python
# BFS template
def bfs_template(start):
    from collections import deque
    
    queue = deque([start])
    visited = set([start])
    
    while queue:
        node = queue.popleft()
        
        # Process node
        print(node)
        
        # Add neighbors
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Level order traversal
def level_order_traversal(root):
    if not root:
        return []
    
    from collections import deque
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

# Implement stack using queues
class StackUsingQueues:
    def __init__(self):
        from collections import deque
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self):
        return self.q1.popleft()
    
    def top(self):
        return self.q1[0]
    
    def empty(self):
        return len(self.q1) == 0
```

---

# Part II: Algorithm Patterns

## 1. Two Pointers

### When to Use
- Working with sorted arrays or linked lists
- Need to find pairs/triplets that satisfy certain conditions
- Optimizing brute force O(n²) solutions to O(n)
- Comparing elements from opposite ends or different positions

### Where to Use
- Array/String problems involving pairs
- Removing duplicates
- Palindrome checking
- Container with most water problems

### Pattern Template
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Process current elements
        if condition_met:
            # Do something
            left += 1
            right -= 1
        elif need_larger_sum:
            left += 1
        else:
            right -= 1
    
    return result
```

### LeetCode Example: Two Sum II (LC 167)
**Problem:** Given a sorted array, find two numbers that add up to target.

**Note:** This problem requires **1-indexed** return values (not 0-indexed).

**Brute Force - O(n²)**
```python
def twoSum_bruteforce(numbers, target):
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if numbers[i] + numbers[j] == target:
                return [i + 1, j + 1]  # +1 for 1-indexed result
    return []
```

**Optimal - O(n)**
```python
def twoSum(numbers, target):
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # +1 for 1-indexed result
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### LeetCode Example: Valid Palindrome (LC 125)

**Brute Force - O(n)**
```python
def isPalindrome_bruteforce(s):
    # Clean string
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

**Optimal - O(n), O(1) space**
```python
def isPalindrome(s):
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

### LeetCode Example: 3Sum (LC 15)

**Brute Force - O(n³)**
```python
def threeSum_bruteforce(nums):
    result = []
    n = len(nums)
    nums.sort()
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    triplet = [nums[i], nums[j], nums[k]]
                    if triplet not in result:
                        result.append(triplet)
    
    return result
```

**Optimal - O(n²)**
```python
def threeSum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second and third numbers
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

---

## 2. Sliding Window

### When to Use
- Finding subarrays/substrings that satisfy certain conditions
- Working with contiguous sequences
- Need to track a window of elements
- Optimizing from O(n²) to O(n)

### Where to Use
- Maximum/minimum subarray problems
- Substring problems
- Fixed or variable window size problems
- Problems with "contiguous" in description

### Pattern Template
```python
# Fixed size window
def sliding_window_fixed(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Variable size window
def sliding_window_variable(arr, target):
    left = 0
    window_sum = 0
    result = float('inf')
    
    for right in range(len(arr)):
        window_sum += arr[right]
        
        while window_sum >= target:
            result = min(result, right - left + 1)
            window_sum -= arr[left]
            left += 1
    
    return result if result != float('inf') else 0
```

### LeetCode Example: Maximum Average Subarray (LC 643)

**Brute Force - O(n*k)**
```python
def findMaxAverage_bruteforce(nums, k):
    max_avg = float('-inf')
    
    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i + k])
        max_avg = max(max_avg, current_sum / k)
    
    return max_avg
```

**Optimal - O(n)**
```python
def findMaxAverage(nums, k):
    # Calculate first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum / k
```

### LeetCode Example: Longest Substring Without Repeating Characters (LC 3)

**Brute Force - O(n³)**
```python
def lengthOfLongestSubstring_bruteforce(s):
    def has_duplicates(substring):
        return len(substring) != len(set(substring))
    
    max_len = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if not has_duplicates(s[i:j]):
                max_len = max(max_len, j - i)
    
    return max_len
```

**Optimal - O(n)**
```python
def lengthOfLongestSubstring(s):
    char_set = set()
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        # Shrink window if duplicate found
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### LeetCode Example: Minimum Window Substring (LC 76)

**Brute Force - O(n²)**
```python
def minWindow_bruteforce(s, t):
    def contains_all(window, target):
        from collections import Counter
        window_count = Counter(window)
        target_count = Counter(target)
        for char, count in target_count.items():
            if window_count[char] < count:
                return False
        return True
    
    min_len = float('inf')
    result = ""
    
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if contains_all(s[i:j], t):
                if j - i < min_len:
                    min_len = j - i
                    result = s[i:j]
    
    return result
```

**Optimal - O(n)**
```python
def minWindow(s, t):
    from collections import Counter
    
    if not t or not s:
        return ""
    
    # Count characters in t
    dict_t = Counter(t)
    required = len(dict_t)  # Number of unique chars in t that must be in window
    
    # Sliding window
    left, right = 0, 0
    formed = 0  # Number of unique chars in window with desired frequency
    window_counts = {}
    
    # (window length, left, right)
    ans = float('inf'), None, None
    
    while right < len(s):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1
        
        # Try to contract the window
        while left <= right and formed == required:
            char = s[left]
            
            # Update result if this window is smaller
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

---

## 3. Fast & Slow Pointers

### When to Use
- Detecting cycles in linked lists
- Finding middle element
- Detecting patterns in sequences
- Problems involving linked list manipulation

### Where to Use
- Cycle detection
- Finding middle of linked list
- Happy number problems
- Palindrome linked list

### Pattern Template
```python
def fast_slow_pointers(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:  # Cycle detected
            return True
    
    return False
```

### LeetCode Example: Linked List Cycle (LC 141)

**Brute Force - O(n) time, O(n) space**
```python
def hasCycle_bruteforce(head):
    visited = set()
    current = head
    
    while current:
        if current in visited:
            return True
        visited.add(current)
        current = current.next
    
    return False
```

**Optimal - O(n) time, O(1) space**
```python
def hasCycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True
```

### LeetCode Example: Middle of Linked List (LC 876)

**Brute Force - O(n) with two passes**
```python
def middleNode_bruteforce(head):
    # First pass: count length
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Second pass: find middle
    mid = length // 2
    current = head
    for _ in range(mid):
        current = current.next
    
    return current
```

**Optimal - O(n) with one pass**
```python
def middleNode(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

### LeetCode Example: Happy Number (LC 202)

**Brute Force - O(log n) time, O(log n) space**
```python
def isHappy_bruteforce(n):
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    
    return n == 1
```

**Optimal - O(log n) time, O(1) space**
```python
def isHappy(n):
    def get_next(num):
        return sum(int(digit) ** 2 for digit in str(num))
    
    # Treat the sequence as a linked list - use Floyd's cycle detection
    slow = n
    fast = get_next(n)
    
    while fast != 1 and slow != fast:
        slow = get_next(slow)  # Move 1 step
        fast = get_next(get_next(fast))  # Move 2 steps
    
    return fast == 1
```

---

## 4. Binary Search

### When to Use
- Searching in sorted arrays
- Finding boundaries or specific conditions
- Optimizing from O(n) to O(log n)
- "Find minimum/maximum that satisfies condition"

### Where to Use
- Sorted arrays
- Search space can be divided into two halves
- Finding peak elements
- Rotated sorted arrays

### Pattern Template
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Use left + (right - left) // 2 to avoid overflow in other languages
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Finding leftmost position
def binary_search_left(arr, target):
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left
```

### LeetCode Example: Binary Search (LC 704)

**Brute Force - O(n)**
```python
def search_bruteforce(nums, target):
    for i in range(len(nums)):
        if nums[i] == target:
            return i
    return -1
```

**Optimal - O(log n)**
```python
def search(nums, target):
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
```

### LeetCode Example: Find First and Last Position (LC 34)

**Brute Force - O(n)**
```python
def searchRange_bruteforce(nums, target):
    result = [-1, -1]
    
    for i in range(len(nums)):
        if nums[i] == target:
            if result[0] == -1:
                result[0] = i
            result[1] = i
    
    return result
```

**Optimal - O(log n)**
```python
def searchRange(nums, target):
    def find_left(nums, target):
        # Find leftmost position where target could be inserted
        left, right = 0, len(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def find_right(nums, target):
        # Find rightmost position where target could be inserted
        left, right = 0, len(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        
        return left - 1  # -1 to get last occurrence position
    
    left_idx = find_left(nums, target)
    right_idx = find_right(nums, target)
    
    # Check if target exists (left_idx will be at target's position if it exists)
    if left_idx <= right_idx and left_idx < len(nums) and nums[left_idx] == target:
        return [left_idx, right_idx]
    
    return [-1, -1]
```

### LeetCode Example: Search in Rotated Sorted Array (LC 33)

**Brute Force - O(n)**
```python
def search_rotated_bruteforce(nums, target):
    for i in range(len(nums)):
        if nums[i] == target:
            return i
    return -1
```

**Optimal - O(log n)**
```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # At least one half must be sorted - determine which one
        if nums[left] <= nums[mid]:  # Left half is sorted
            # Check if target is in the sorted left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            # Check if target is in the sorted right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

---

## 5. Merge Intervals

### When to Use
- Overlapping intervals problems
- Scheduling problems
- Time range merging
- Problems with start and end points

### Where to Use
- Meeting rooms
- Interval overlapping
- Interval insertion
- Minimum meeting rooms required

### Pattern Template
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:  # Overlapping
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    return merged
```

### LeetCode Example: Merge Intervals (LC 56)

**Brute Force - O(n²)**
```python
def merge_bruteforce(intervals):
    result = []
    used = [False] * len(intervals)
    
    for i in range(len(intervals)):
        if used[i]:
            continue
        
        current = intervals[i][:]
        used[i] = True
        
        # Keep merging with overlapping intervals
        changed = True
        while changed:
            changed = False
            for j in range(len(intervals)):
                if not used[j]:
                    if intervals[j][0] <= current[1]:
                        current[0] = min(current[0], intervals[j][0])
                        current[1] = max(current[1], intervals[j][1])
                        used[j] = True
                        changed = True
        
        result.append(current)
    
    return sorted(result)
```

**Optimal - O(n log n)**
```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged
```

### LeetCode Example: Insert Interval (LC 57)

**Brute Force - O(n)**
```python
def insert_bruteforce(intervals, newInterval):
    intervals.append(newInterval)
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged
```

**Optimal - O(n)**
```python
def insert(intervals, newInterval):
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before newInterval
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    result.append(newInterval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result
```

### LeetCode Example: Meeting Rooms II (LC 253)

**Brute Force - O(n²)**
```python
def minMeetingRooms_bruteforce(intervals):
    if not intervals:
        return 0
    
    max_rooms = 0
    
    for i in range(len(intervals)):
        rooms_needed = 1
        for j in range(len(intervals)):
            if i != j:
                # Check if intervals overlap
                if intervals[i][0] < intervals[j][1] and intervals[j][0] < intervals[i][1]:
                    rooms_needed += 1
        max_rooms = max(max_rooms, rooms_needed)
    
    return max_rooms
```

**Optimal - O(n log n)**
```python
def minMeetingRooms(intervals):
    if not intervals:
        return 0
    
    import heapq
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times
    heap = []
    heapq.heappush(heap, intervals[0][1])
    
    for interval in intervals[1:]:
        # If earliest ending meeting has finished, reuse room
        if interval[0] >= heap[0]:
            heapq.heappop(heap)
        
        heapq.heappush(heap, interval[1])
    
    return len(heap)
```

---

## 6. In-place Reversal of Linked List

### When to Use
- Reversing linked list or part of it
- Avoiding extra space for linked list problems
- Problems requiring reversal in groups

### Where to Use
- Reverse linked list
- Reverse in groups
- Palindrome linked list
- Reorder list

### Pattern Template
```python
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

### LeetCode Example: Reverse Linked List (LC 206)

**Brute Force - O(n) time, O(n) space**
```python
def reverseList_bruteforce(head):
    # Store values in array
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    # Rebuild list in reverse
    if not values:
        return None
    
    new_head = ListNode(values[-1])
    current = new_head
    for i in range(len(values) - 2, -1, -1):
        current.next = ListNode(values[i])
        current = current.next
    
    return new_head
```

**Optimal - O(n) time, O(1) space**
```python
def reverseList(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

### LeetCode Example: Reverse Linked List II (LC 92)

**Brute Force - O(n) time, O(n) space**
```python
def reverseBetween_bruteforce(head, left, right):
    # Store all values
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    # Reverse the subarray
    values[left - 1:right] = reversed(values[left - 1:right])
    
    # Rebuild list
    dummy = ListNode(0)
    current = dummy
    for val in values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next
```

**Optimal - O(n) time, O(1) space**
```python
def reverseBetween(head, left, right):
    if not head or left == right:
        return head
    
    # Use dummy node to handle edge case where left=1
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move to position before left
    for _ in range(left - 1):
        prev = prev.next
    
    # Reverse the sublist by repeatedly moving next node to front
    current = prev.next
    for _ in range(right - left):
        next_node = current.next
        current.next = next_node.next  # Skip next_node
        next_node.next = prev.next  # Insert at front
        prev.next = next_node  # Update front
    
    return dummy.next
```

### LeetCode Example: Reverse Nodes in k-Group (LC 25)

**Brute Force - O(n) time, O(n) space**
```python
def reverseKGroup_bruteforce(head, k):
    # Collect all values
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    # Reverse in groups of k
    for i in range(0, len(values) - k + 1, k):
        values[i:i + k] = reversed(values[i:i + k])
    
    # Rebuild list
    dummy = ListNode(0)
    current = dummy
    for val in values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next
```

**Optimal - O(n) time, O(1) space**
```python
def reverseKGroup(head, k):
    def get_length(node):
        count = 0
        while node:
            count += 1
            node = node.next
        return count
    
    def reverse_k_nodes(head, k):
        prev = None
        current = head
        for _ in range(k):
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev, head, current
    
    length = get_length(head)
    dummy = ListNode(0)
    dummy.next = head
    prev_group = dummy
    
    while length >= k:
        group_start = prev_group.next
        new_head, new_tail, next_group = reverse_k_nodes(group_start, k)
        
        prev_group.next = new_head
        new_tail.next = next_group
        prev_group = new_tail
        
        length -= k
    
    return dummy.next
```

---

## 7. Tree BFS

### When to Use
- Level-order traversal
- Finding shortest path in trees
- Level-by-level processing
- Finding minimum depth

### Where to Use
- Level order traversal
- Zigzag traversal
- Connect level order siblings
- Tree right side view

### Pattern Template
```python
from collections import deque

def bfs_tree(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### LeetCode Example: Binary Tree Level Order Traversal (LC 102)

**Brute Force - O(n²) using DFS**
```python
def levelOrder_bruteforce(root):
    if not root:
        return []
    
    def get_height(node):
        if not node:
            return 0
        return 1 + max(get_height(node.left), get_height(node.right))
    
    def get_level(node, level, current_level, result):
        if not node:
            return
        if current_level == level:
            result.append(node.val)
        else:
            get_level(node.left, level, current_level + 1, result)
            get_level(node.right, level, current_level + 1, result)
    
    height = get_height(root)
    result = []
    
    for level in range(height):
        level_nodes = []
        get_level(root, level, 0, level_nodes)
        result.append(level_nodes)
    
    return result
```

**Optimal - O(n) using BFS**
```python
def levelOrder(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### LeetCode Example: Binary Tree Zigzag Level Order (LC 103)

**Brute Force - O(n) with reversing**
```python
def zigzagLevelOrder_bruteforce(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        if not left_to_right:
            level.reverse()
        
        result.append(level)
        left_to_right = not left_to_right
    
    return result
```

**Optimal - O(n) with deque**
```python
def zigzagLevelOrder(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level))
        left_to_right = not left_to_right
    
    return result
```

### LeetCode Example: Binary Tree Right Side View (LC 199)

**Brute Force - O(n) collecting all levels**
```python
def rightSideView_bruteforce(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level[-1])  # Take rightmost of each level
    
    return result
```

**Optimal - O(n) taking only rightmost**
```python
def rightSideView(root):
    if not root:
        return []
    
    from collections import deque
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # If it's the last node in the level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

---

## 8. Tree DFS

### When to Use
- Need to explore all paths
- Backtracking problems
- Finding paths from root to leaf
- Tree traversal (preorder, inorder, postorder)

### Where to Use
- Path sum problems
- Validate BST
- Tree diameter
- Lowest common ancestor

### Pattern Template
```python
def dfs_tree(root):
    def dfs(node):
        if not node:
            return
        
        # Process current node (preorder)
        print(node.val)
        
        dfs(node.left)
        dfs(node.right)
    
    dfs(root)

# With return value
def dfs_with_return(root):
    def dfs(node):
        if not node:
            return base_case_value
        
        left_result = dfs(node.left)
        right_result = dfs(node.right)
        
        return combine(node.val, left_result, right_result)
    
    return dfs(root)
```

### LeetCode Example: Maximum Depth of Binary Tree (LC 104)

**Brute Force - O(n) using BFS**
```python
def maxDepth_bruteforce(root):
    if not root:
        return 0
    
    from collections import deque
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth
```

**Optimal - O(n) using DFS**
```python
def maxDepth(root):
    if not root:
        return 0
    
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    
    return 1 + max(left_depth, right_depth)
```

### LeetCode Example: Path Sum (LC 112)

**Brute Force - O(n) collecting all paths**
```python
def hasPathSum_bruteforce(root, targetSum):
    if not root:
        return False
    
    def get_all_paths(node, current_path, all_paths):
        if not node:
            return
        
        current_path.append(node.val)
        
        if not node.left and not node.right:
            all_paths.append(sum(current_path))
        
        get_all_paths(node.left, current_path[:], all_paths)
        get_all_paths(node.right, current_path[:], all_paths)
    
    all_paths = []
    get_all_paths(root, [], all_paths)
    
    return targetSum in all_paths
```

**Optimal - O(n) with early termination**
```python
def hasPathSum(root, targetSum):
    if not root:
        return False
    
    # Leaf node
    if not root.left and not root.right:
        return root.val == targetSum
    
    # Recursively check left and right subtrees
    return (hasPathSum(root.left, targetSum - root.val) or
            hasPathSum(root.right, targetSum - root.val))
```

### LeetCode Example: Diameter of Binary Tree (LC 543)

**Brute Force - O(n²)**
```python
def diameterOfBinaryTree_bruteforce(root):
    def height(node):
        if not node:
            return 0
        return 1 + max(height(node.left), height(node.right))
    
    def diameter(node):
        if not node:
            return 0
        
        # Diameter through this node
        left_height = height(node.left)
        right_height = height(node.right)
        diameter_through_node = left_height + right_height
        
        # Diameter not through this node
        left_diameter = diameter(node.left)
        right_diameter = diameter(node.right)
        
        return max(diameter_through_node, left_diameter, right_diameter)
    
    return diameter(root)
```

**Optimal - O(n)**
```python
def diameterOfBinaryTree(root):
    def dfs(node):
        if not node:
            return 0, 0  # (height, diameter)
        
        left_height, left_diameter = dfs(node.left)
        right_height, right_diameter = dfs(node.right)
        
        height = 1 + max(left_height, right_height)
        diameter = max(left_height + right_height, left_diameter, right_diameter)
        
        return height, diameter
    
    _, diameter = dfs(root)
    return diameter
```

---

## 9. Two Heaps

### When to Use
- Finding median in data stream
- Sliding window median
- Problems requiring min and max simultaneously
- Balancing two halves of data

### Where to Use
- Median finder
- IPO problem
- Sliding window median
- Schedule tasks

### Pattern Template
```python
import heapq

class MedianFinder:
    def __init__(self):
        # Python heapq is min heap only, so negate values for max heap behavior
        self.max_heap = []  # Lower half (store negative values)
        self.min_heap = []  # Upper half (normal positive values)
    
    def addNum(self, num):
        # Add to max heap (lower half)
        heapq.heappush(self.max_heap, -num)
        
        # Balance: move largest from lower to upper
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # Maintain size: max_heap can have at most 1 more element
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2
```

### LeetCode Example: Find Median from Data Stream (LC 295)

**Brute Force - O(n log n) per median query**
```python
class MedianFinder_bruteforce:
    def __init__(self):
        self.nums = []
    
    def addNum(self, num):
        self.nums.append(num)
    
    def findMedian(self):
        sorted_nums = sorted(self.nums)
        n = len(sorted_nums)
        
        if n % 2 == 1:
            return sorted_nums[n // 2]
        else:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
```

**Optimal - O(log n) per add, O(1) per median**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.max_heap = []  # Lower half (use negative values)
        self.min_heap = []  # Upper half
    
    def addNum(self, num):
        # Add to max heap first
        heapq.heappush(self.max_heap, -num)
        
        # Move largest from max heap to min heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # Balance sizes
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

### LeetCode Example: Sliding Window Median (LC 480)

**Brute Force - O(n * k log k)**
```python
def medianSlidingWindow_bruteforce(nums, k):
    def find_median(window):
        sorted_window = sorted(window)
        if k % 2 == 1:
            return float(sorted_window[k // 2])
        else:
            return (sorted_window[k // 2 - 1] + sorted_window[k // 2]) / 2.0
    
    result = []
    for i in range(len(nums) - k + 1):
        result.append(find_median(nums[i:i + k]))
    
    return result
```

**Optimal - O(n log k) using two heaps**
```python
from heapq import heappush, heappop
import heapq

def medianSlidingWindow(nums, k):
    def get_median():
        if k % 2 == 1:
            return float(-max_heap[0])
        return (-max_heap[0] + min_heap[0]) / 2.0
    
    max_heap = []  # Lower half
    min_heap = []  # Upper half
    result = []
    
    # Helper to remove element (uses heapq private methods for efficiency)
    # Note: In production, consider using a lazy deletion approach instead
    def remove_element(heap, val):
        idx = heap.index(val)
        heap[idx] = heap[-1]
        heap.pop()
        if idx < len(heap):
            heapq._siftup(heap, idx)
            heapq._siftdown(heap, 0, idx)
    
    # Initialize first window
    for i in range(k):
        heappush(max_heap, -nums[i])
    
    for _ in range(k // 2):
        heappush(min_heap, -heappop(max_heap))
    
    result.append(get_median())
    
    # Slide window
    for i in range(k, len(nums)):
        # Remove outgoing element
        outgoing = nums[i - k]
        balance = -1 if outgoing <= -max_heap[0] else 1
        
        if outgoing <= -max_heap[0]:
            remove_element(max_heap, -outgoing)
        else:
            remove_element(min_heap, outgoing)
        
        # Add incoming element
        if not max_heap or nums[i] <= -max_heap[0]:
            heappush(max_heap, -nums[i])
            balance += 1
        else:
            heappush(min_heap, nums[i])
            balance -= 1
        
        # Rebalance
        if balance < 0:
            heappush(max_heap, -heappop(min_heap))
        elif balance > 0:
            heappush(min_heap, -heappop(max_heap))
        
        result.append(get_median())
    
    return result
```

---

## 10. Subsets/Backtracking

### When to Use
- Generate all possible combinations
- Permutations problems
- Combination sum problems
- N-Queens, Sudoku solver

### Where to Use
- Generate subsets/powersets
- Permutations
- Combinations
- Constraint satisfaction problems

### Pattern Template
```python
def backtrack_subsets(nums):
    result = []
    
    def backtrack(start, current):
        result.append(current[:])  # Make a copy ([:]) to avoid reference issues
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()  # Backtrack
    
    backtrack(0, [])
    return result

def backtrack_permutations(nums):
    result = []
    
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        
        for i in range(len(remaining)):
            backtrack(current + [remaining[i]], 
                     remaining[:i] + remaining[i+1:])
    
    backtrack([], nums)
    return result
```

### LeetCode Example: Subsets (LC 78)

**Brute Force - O(2^n * n) using bit manipulation**
```python
def subsets_bruteforce(nums):
    n = len(nums)
    result = []
    
    # Generate all 2^n combinations
    for i in range(2 ** n):
        subset = []
        for j in range(n):
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    
    return result
```

**Optimal - O(2^n * n) using backtracking**
```python
def subsets(nums):
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

### LeetCode Example: Permutations (LC 46)

**Brute Force - O(n! * n²) generating and checking**
```python
def permute_bruteforce(nums):
    from itertools import permutations
    return [list(p) for p in permutations(nums)]
```

**Optimal - O(n! * n) using backtracking**
```python
def permute(nums):
    result = []
    
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        
        for i in range(len(remaining)):
            backtrack(current + [remaining[i]], 
                     remaining[:i] + remaining[i+1:])
    
    backtrack([], nums)
    return result

# Alternative approach with swapping
def permute_swap(nums):
    result = []
    
    def backtrack(first):
        if first == len(nums):
            result.append(nums[:])
            return
        
        for i in range(first, len(nums)):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first + 1)
            nums[first], nums[i] = nums[i], nums[first]
    
    backtrack(0)
    return result
```

### LeetCode Example: Combination Sum (LC 39)

**Brute Force - Exponential with duplicates**
```python
def combinationSum_bruteforce(candidates, target):
    result = []
    
    def backtrack(remain, start, current):
        if remain < 0:
            return
        if remain == 0:
            result.append(sorted(current[:]))
            return
        
        for num in candidates:
            current.append(num)
            backtrack(remain - num, start, current)
            current.pop()
    
    backtrack(target, 0, [])
    
    # Remove duplicates
    return [list(x) for x in set(tuple(x) for x in result)]
```

**Optimal - O(N^(T/M)) where T=target, M=min candidate**
```python
def combinationSum(candidates, target):
    result = []
    
    def backtrack(remain, start, current):
        if remain == 0:
            result.append(current[:])
            return
        if remain < 0:
            return
        
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            # Not i+1 because we can reuse same element
            backtrack(remain - candidates[i], i, current)
            current.pop()
    
    backtrack(target, 0, [])
    return result
```

---

## 11. Top K Elements

### When to Use
- Finding K largest/smallest elements
- K most frequent elements
- Problems with "top K", "closest K", "K pairs"

### Where to Use
- Kth largest element
- K closest points
- Top K frequent words
- K pairs with smallest sum

### Pattern Template
```python
import heapq

def top_k_elements(nums, k):
    # For K largest, use min heap
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap

def top_k_frequent(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # Use heap with frequency
    return heapq.nlargest(k, count.keys(), key=count.get)
```

### LeetCode Example: Kth Largest Element (LC 215)

**Brute Force - O(n log n)**
```python
def findKthLargest_bruteforce(nums, k):
    nums.sort(reverse=True)
    return nums[k - 1]
```

**Optimal - O(n log k) using heap**
```python
import heapq

def findKthLargest(nums, k):
    # Use min heap of size k
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0]

# Even better: O(n) average, O(n²) worst case using quickselect
def findKthLargest_quickselect(nums, k):
    def partition(left, right, pivot_idx):
        pivot = nums[pivot_idx]
        # Move pivot to end for partitioning
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        # Move pivot to final position
        nums[right], nums[store_idx] = nums[store_idx], nums[right]
        return store_idx
    
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        pivot_idx = left + (right - left) // 2
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return select(left, pivot_idx - 1, k_smallest)
        else:
            return select(pivot_idx + 1, right, k_smallest)
    
    # Convert kth largest to (n-k)th smallest (0-indexed)
    return select(0, len(nums) - 1, len(nums) - k)
```

### LeetCode Example: Top K Frequent Elements (LC 347)

**Brute Force - O(n log n)**
```python
def topKFrequent_bruteforce(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # Sort by frequency
    sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in sorted_items[:k]]
```

**Optimal - O(n log k) using heap**
```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    
    # Use heap to get top k
    return heapq.nlargest(k, count.keys(), key=count.get)

# Alternative: O(n) using bucket sort
def topKFrequent_bucket(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # Bucket sort: index = frequency, value = list of numbers with that frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result
```

### LeetCode Example: K Closest Points to Origin (LC 973)

**Brute Force - O(n log n)**
```python
def kClosest_bruteforce(points, k):
    def distance(point):
        return point[0] ** 2 + point[1] ** 2
    
    points.sort(key=distance)
    return points[:k]
```

**Optimal - O(n log k)**
```python
import heapq

def kClosest(points, k):
    def distance(point):
        return point[0] ** 2 + point[1] ** 2
    
    # Use max heap of size k (negate distances)
    max_heap = []
    
    for point in points:
        dist = distance(point)
        heapq.heappush(max_heap, (-dist, point))
        
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    return [point for (_, point) in max_heap]

# Alternative: O(n) average using quickselect
def kClosest_quickselect(points, k):
    def distance(point):
        return point[0] ** 2 + point[1] ** 2
    
    distances = [distance(p) for p in points]
    
    def partition(left, right, pivot_idx):
        pivot_dist = distances[pivot_idx]
        distances[pivot_idx], distances[right] = distances[right], distances[pivot_idx]
        points[pivot_idx], points[right] = points[right], points[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if distances[i] < pivot_dist:
                distances[store_idx], distances[i] = distances[i], distances[store_idx]
                points[store_idx], points[i] = points[i], points[store_idx]
                store_idx += 1
        
        distances[right], distances[store_idx] = distances[store_idx], distances[right]
        points[right], points[store_idx] = points[store_idx], points[right]
        return store_idx
    
    def select(left, right, k):
        if left == right:
            return
        
        pivot_idx = left + (right - left) // 2
        pivot_idx = partition(left, right, pivot_idx)
        
        if k == pivot_idx:
            return
        elif k < pivot_idx:
            select(left, pivot_idx - 1, k)
        else:
            select(pivot_idx + 1, right, k)
    
    select(0, len(points) - 1, k - 1)
    return points[:k]
```

---

## 12. Dynamic Programming

### When to Use
- Optimization problems (min/max)
- Counting problems
- Problems with overlapping subproblems
- Problems asking for "all possible ways"

### Where to Use
- Fibonacci-like sequences
- Knapsack problems
- Longest increasing subsequence
- Edit distance
- Matrix chain multiplication

### Pattern Template
```python
# Top-down (Memoization)
def dp_topdown(n):
    memo = {}
    
    def helper(state):
        if state in memo:
            return memo[state]
        
        if base_case:
            return base_value
        
        # Recursive relation
        result = combine(helper(subproblem1), helper(subproblem2))
        memo[state] = result
        return result
    
    return helper(n)

# Bottom-up (Tabulation)
def dp_bottomup(n):
    dp = [0] * (n + 1)
    dp[0] = base_value
    
    for i in range(1, n + 1):
        dp[i] = combine(dp[i-1], dp[i-2], ...)
    
    return dp[n]
```

### LeetCode Example: Climbing Stairs (LC 70)

**Brute Force - O(2^n) recursion**
```python
def climbStairs_bruteforce(n):
    if n <= 2:
        return n
    
    return climbStairs_bruteforce(n - 1) + climbStairs_bruteforce(n - 2)
```

**Optimal - O(n) time, O(n) space**
```python
def climbStairs(n):
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Space optimized - O(1) space
def climbStairs_optimized(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### LeetCode Example: Coin Change (LC 322)

**Brute Force - Exponential**
```python
def coinChange_bruteforce(coins, amount):
    def helper(remaining):
        if remaining < 0:
            return float('inf')
        if remaining == 0:
            return 0
        
        min_coins = float('inf')
        for coin in coins:
            min_coins = min(min_coins, 1 + helper(remaining - coin))
        
        return min_coins
    
    result = helper(amount)
    return result if result != float('inf') else -1
```

**Optimal - O(amount * n) time, O(amount) space**
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### LeetCode Example: Longest Increasing Subsequence (LC 300)

**Brute Force - O(2^n)**
```python
def lengthOfLIS_bruteforce(nums):
    def helper(idx, prev):
        if idx == len(nums):
            return 0
        
        # Skip current
        taken = 0
        if nums[idx] > prev:
            taken = 1 + helper(idx + 1, nums[idx])
        
        not_taken = helper(idx + 1, prev)
        
        return max(taken, not_taken)
    
    return helper(0, float('-inf'))
```

**Optimal - O(n²) DP**
```python
def lengthOfLIS(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Even better: O(n log n) using binary search
def lengthOfLIS_optimal(nums):
    from bisect import bisect_left
    
    sub = []  # Maintains smallest tail for each length
    
    for num in nums:
        # bisect_left finds leftmost position to insert num
        pos = bisect_left(sub, num)
        
        if pos == len(sub):
            sub.append(num)
        else:
            sub[pos] = num
    
    return len(sub)
```

---

## 13. Monotonic Stack

### When to Use
- Next greater/smaller element problems
- Finding spans or ranges
- Problems involving "nearest" larger/smaller element
- Stock span problems

### Where to Use
- Next greater element
- Daily temperatures
- Largest rectangle in histogram
- Trapping rain water

### Pattern Template
```python
def monotonic_stack(arr):
    stack = []
    result = []
    
    for i in range(len(arr)):
        # Maintain decreasing stack for next greater element
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]  # arr[i] is next greater for arr[idx]
        
        stack.append(i)
    
    return result
```

### LeetCode Example: Next Greater Element I (LC 496)

**Brute Force - O(n*m)**
```python
def nextGreaterElement_bruteforce(nums1, nums2):
    result = []
    
    for num in nums1:
        # Find num in nums2
        idx = nums2.index(num)
        
        # Find next greater
        next_greater = -1
        for j in range(idx + 1, len(nums2)):
            if nums2[j] > num:
                next_greater = nums2[j]
                break
        
        result.append(next_greater)
    
    return result
```

**Optimal - O(n+m)**
```python
def nextGreaterElement(nums1, nums2):
    # Build next greater map for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # For remaining elements in stack
    while stack:
        next_greater[stack.pop()] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]
```

### LeetCode Example: Daily Temperatures (LC 739)

**Brute Force - O(n²)**
```python
def dailyTemperatures_bruteforce(temperatures):
    n = len(temperatures)
    result = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            if temperatures[j] > temperatures[i]:
                result[i] = j - i
                break
    
    return result
```

**Optimal - O(n)**
```python
def dailyTemperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []
    
    for i in range(n):
        # While current temp is warmer than stack top
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        
        stack.append(i)
    
    return result
```

### LeetCode Example: Largest Rectangle in Histogram (LC 84)

**Brute Force - O(n²)**
```python
def largestRectangleArea_bruteforce(heights):
    max_area = 0
    
    for i in range(len(heights)):
        min_height = heights[i]
        for j in range(i, len(heights)):
            min_height = min(min_height, heights[j])
            area = min_height * (j - i + 1)
            max_area = max(max_area, area)
    
    return max_area
```

**Optimal - O(n)**
```python
def largestRectangleArea(heights):
    stack = []  # Stack stores indices
    max_area = 0
    heights.append(0)  # Sentinel to flush remaining bars from stack
    
    for i in range(len(heights)):
        # Pop bars that are taller than current (they can't extend further)
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            # Width: if stack empty, bar extends to beginning; else to element after stack top
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        stack.append(i)
    
    heights.pop()  # Remove sentinel
    return max_area
```

---

## 14. Prefix Sum

### When to Use
- Range sum queries
- Subarray sum problems
- Finding subarrays with specific sum properties
- Optimization from O(n) per query to O(1)

### Where to Use
- Subarray sum equals K
- Continuous subarray sum
- Range sum queries
- Product of array except self

### Pattern Template
```python
def prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    
    # Query sum from index i to j (inclusive)
    def range_sum(i, j):
        return prefix[j + 1] - prefix[i]
    
    return prefix

# With hashmap for subarray problems
def subarray_with_sum(arr, target):
    prefix_sum = 0
    sum_count = {0: 1}
    count = 0
    
    for num in arr:
        prefix_sum += num
        
        if prefix_sum - target in sum_count:
            count += sum_count[prefix_sum - target]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count
```

### LeetCode Example: Range Sum Query (LC 303)

**Brute Force - O(n) per query**
```python
class NumArray_bruteforce:
    def __init__(self, nums):
        self.nums = nums
    
    def sumRange(self, left, right):
        return sum(self.nums[left:right + 1])
```

**Optimal - O(1) per query**
```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

### LeetCode Example: Subarray Sum Equals K (LC 560)

**Brute Force - O(n²)**
```python
def subarraySum_bruteforce(nums, k):
    count = 0
    
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            if current_sum == k:
                count += 1
    
    return count
```

**Optimal - O(n)**
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Initialize with 0 to handle subarrays starting from index 0
    
    for num in nums:
        prefix_sum += num
        
        # If (prefix_sum - k) exists, there's a subarray ending here with sum k
        # Because prefix_sum - (prefix_sum - k) = k
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count
```

### LeetCode Example: Product of Array Except Self (LC 238)

**Brute Force - O(n²)**
```python
def productExceptSelf_bruteforce(nums):
    n = len(nums)
    result = []
    
    for i in range(n):
        product = 1
        for j in range(n):
            if i != j:
                product *= nums[j]
        result.append(product)
    
    return result
```

**Optimal - O(n) time, O(1) extra space (excluding output array)**
```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    
    # First pass: calculate product of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Second pass: multiply by product of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result
```

---

## 15. Union Find

### When to Use
- Connected components problems
- Detecting cycles in undirected graphs
- Dynamic connectivity
- Grouping/clustering problems

### Where to Use
- Number of islands
- Friend circles
- Redundant connections
- Accounts merge

### Pattern Template
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        # Path compression: make every node point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank: attach smaller tree under larger tree
        # This keeps the tree balanced and maintains O(α(n)) complexity
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### LeetCode Example: Number of Provinces (LC 547)

**Brute Force - O(n²) DFS**
```python
def findCircleNum_bruteforce(isConnected):
    n = len(isConnected)
    visited = [False] * n
    
    def dfs(i):
        visited[i] = True
        for j in range(n):
            if isConnected[i][j] == 1 and not visited[j]:
                dfs(j)
    
    provinces = 0
    for i in range(n):
        if not visited[i]:
            dfs(i)
            provinces += 1
    
    return provinces
```

**Optimal - O(n²) with Union Find**
```python
def findCircleNum(isConnected):
    n = len(isConnected)
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                uf.union(i, j)
    
    return uf.count

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
```

### LeetCode Example: Redundant Connection (LC 684)

**Brute Force - O(n²) trying to remove each edge**
```python
def findRedundantConnection_bruteforce(edges):
    def has_cycle_without_edge(skip_idx):
        adj = {}
        for i, (u, v) in enumerate(edges):
            if i == skip_idx:
                continue
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
        
        visited = set()
        def dfs(node, parent):
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True
                if dfs(neighbor, node):
                    return True
            return False
        
        for node in adj:
            if node not in visited:
                if dfs(node, -1):
                    return True
        return False
    
    for i in range(len(edges) - 1, -1, -1):
        if not has_cycle_without_edge(i):
            return edges[i]
```

**Optimal - O(n) with Union Find**
```python
def findRedundantConnection(edges):
    uf = UnionFind(len(edges) + 1)
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    
    return []

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False
        
        self.parent[root_x] = root_y
        return True
```

---

## 16. Topological Sort

### When to Use
- Ordering tasks with dependencies
- Course schedule problems
- Build systems
- Detecting cycles in directed graphs

### Where to Use
- Course schedule
- Alien dictionary
- Task scheduling
- Build order

### Pattern Template
```python
# Using DFS
def topological_sort_dfs(graph):
    visited = set()
    stack = []
    
    def dfs(node):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        
        stack.append(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return stack[::-1]

# Using BFS (Kahn's algorithm)
def topological_sort_bfs(n, edges):
    from collections import deque
    
    # Build graph and in-degree (number of prerequisites for each node)
    graph = {i: [] for i in range(n)}
    in_degree = [0] * n  # Count of incoming edges
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Start with nodes having no prerequisites
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []
```

### LeetCode Example: Course Schedule (LC 207)

**Brute Force - O(n!) trying all permutations**
```python
def canFinish_bruteforce(numCourses, prerequisites):
    from itertools import permutations
    
    def is_valid_order(order):
        position = {course: i for i, course in enumerate(order)}
        
        for course, prereq in prerequisites:
            if position[prereq] >= position[course]:
                return False
        
        return True
    
    for order in permutations(range(numCourses)):
        if is_valid_order(order):
            return True
    
    return False
```

**Optimal - O(V + E) using DFS**
```python
def canFinish(numCourses, prerequisites):
    # Build adjacency list
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Three-color DFS: 0 = unvisited, 1 = visiting (in current path), 2 = visited
    state = [0] * numCourses
    
    def has_cycle(course):
        if state[course] == 1:  # Currently in recursion stack - cycle detected!
            return True
        if state[course] == 2:  # Already fully processed
            return False
        
        state[course] = 1
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        state[course] = 2
        
        return False
    
    for course in range(numCourses):
        if has_cycle(course):
            return False
    
    return True

# Alternative: BFS (Kahn's algorithm)
def canFinish_bfs(numCourses, prerequisites):
    from collections import deque
    
    graph = {i: [] for i in range(numCourses)}
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    completed = 0
    
    while queue:
        course = queue.popleft()
        completed += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return completed == numCourses
```

### LeetCode Example: Course Schedule II (LC 210)

**Brute Force - Similar to Course Schedule I**

**Optimal - O(V + E) using BFS**
```python
def findOrder(numCourses, prerequisites):
    from collections import deque
    
    graph = {i: [] for i in range(numCourses)}
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []
    
    while queue:
        course = queue.popleft()
        order.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return order if len(order) == numCourses else []
```

---

## 17. Trie

### When to Use
- Prefix matching problems
- Autocomplete systems
- Word search with prefix
- Dictionary implementations

### Where to Use
- Implement Trie
- Word search II
- Autocomplete
- Longest common prefix

### Pattern Template
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### LeetCode Example: Implement Trie (LC 208)

**Brute Force - Using list**
```python
class Trie_bruteforce:
    def __init__(self):
        self.words = []
    
    def insert(self, word):
        if word not in self.words:
            self.words.append(word)
    
    def search(self, word):
        return word in self.words
    
    def startsWith(self, prefix):
        for word in self.words:
            if word.startswith(prefix):
                return True
        return False
```

**Optimal - Using Trie**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### LeetCode Example: Word Search II (LC 212)

**Brute Force - O(m*n*4^L*W) checking each word**
```python
def findWords_bruteforce(board, words):
    def exist(word):
        def dfs(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
                return False
            if board[i][j] != word[k]:
                return False
            
            temp = board[i][j]
            board[i][j] = '#'
            
            found = (dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or 
                    dfs(i, j+1, k+1) or dfs(i, j-1, k+1))
            
            board[i][j] = temp
            return found
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False
    
    return [word for word in words if exist(word)]
```

**Optimal - O(m*n*4^L) using Trie**
```python
def findWords(board, words):
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    # Build Trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    result = []
    m, n = len(board), len(board[0])
    
    def dfs(i, j, node):
        char = board[i][j]
        if char not in node.children:
            return
        
        next_node = node.children[char]
        
        if next_node.word:
            result.append(next_node.word)
            next_node.word = None  # Avoid duplicates
        
        board[i][j] = '#'  # Mark as visited
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#':
                dfs(ni, nj, next_node)
        
        board[i][j] = char  # Restore
        
        # Optimization: prune trie by removing leaf nodes (reduces search space)
        if not next_node.children:
            del node.children[char]
    
    for i in range(m):
        for j in range(n):
            if board[i][j] in root.children:
                dfs(i, j, root)
    
    return result
```

---

## Summary Table

| Pattern | Time Complexity | Space Complexity | Key Indicator |
|---------|----------------|------------------|---------------|
| Two Pointers | O(n) | O(1) | Sorted array, pairs |
| Sliding Window | O(n) | O(k) | Contiguous subarray |
| Fast & Slow | O(n) | O(1) | Cycle detection |
| Binary Search | O(log n) | O(1) | Sorted array |
| Merge Intervals | O(n log n) | O(n) | Overlapping intervals |
| Linked List Reversal | O(n) | O(1) | Reverse in place |
| Tree BFS | O(n) | O(n) | Level-order |
| Tree DFS | O(n) | O(h) | Path problems |
| Two Heaps | O(n log k) | O(k) | Median, top K |
| Backtracking | O(2^n) | O(n) | All combinations |
| Top K | O(n log k) | O(k) | K largest/smallest |
| Dynamic Programming | O(n²) typical | O(n) | Optimization |
| Monotonic Stack | O(n) | O(n) | Next greater |
| Prefix Sum | O(n) | O(n) | Range queries |
| Union Find | O(α(n))* | O(n) | Connected components |
| Topological Sort | O(V+E) | O(V+E) | Dependencies |
| Trie | O(m) | O(n*m) | Prefix matching |

**\* α(n)** is the inverse Ackermann function, which grows extremely slowly (≤ 5 for all practical inputs)

---

## Quick Reference: Pattern Selection

### Array/String Problems
- **Sorted + Find pair**: Two Pointers
- **Contiguous subarray**: Sliding Window
- **Range queries**: Prefix Sum
- **Next greater/smaller**: Monotonic Stack

### Linked List Problems
- **Cycle detection**: Fast & Slow Pointers
- **Reverse**: In-place Reversal
- **Middle element**: Fast & Slow Pointers

### Tree Problems
- **Level-order**: Tree BFS
- **Path problems**: Tree DFS
- **Prefix matching**: Trie

### Graph Problems
- **Connected components**: Union Find
- **Ordering with dependencies**: Topological Sort
- **Cycle detection (directed)**: Topological Sort
- **Cycle detection (undirected)**: Union Find

### Optimization Problems
- **Overlapping subproblems**: Dynamic Programming
- **K largest/smallest**: Top K Elements / Heap
- **Median**: Two Heaps

### Combinatorial Problems
- **All combinations**: Backtracking
- **Permutations**: Backtracking

---

## Tips for Pattern Recognition

1. **Read the problem carefully** - Look for keywords
2. **Identify constraints** - Size limits hint at complexity
3. **Look for sorted/unsorted** - Affects approach choice
4. **Contiguous vs any subset** - Window vs other patterns
5. **Optimization keywords** - "Maximum", "minimum", "longest" → DP or Greedy
6. **"All possible"** - Usually backtracking
7. **K in the problem** - Likely heap/Top K pattern
8. **Interval/range** - Merge intervals or prefix sum

---

**Happy Coding! 🚀**

