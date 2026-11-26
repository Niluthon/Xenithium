"""Classic algorithms and data structures, written plainly with docstrings and doctests."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Optional


def binary_search(items: list[int], target: int) -> int:
    """Return the index of target in the sorted list items, or -1 when it is absent.

    >>> binary_search([1, 3, 5, 7, 9], 7)
    3
    """
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def bubble_sort(items: list[int]) -> list[int]:
    """Sort a list with bubble sort and return it."""
    items = list(items)
    n = len(items)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
                swapped = True
        if not swapped:
            break
    return items


def insertion_sort(items: list[int]) -> list[int]:
    """Sort a list with insertion sort and return it."""
    items = list(items)
    for i in range(1, len(items)):
        key = items[i]
        j = i - 1
        while j >= 0 and items[j] > key:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = key
    return items


def selection_sort(items: list[int]) -> list[int]:
    """Sort a list with selection sort and return it."""
    items = list(items)
    for i in range(len(items)):
        smallest = i
        for j in range(i + 1, len(items)):
            if items[j] < items[smallest]:
                smallest = j
        items[i], items[smallest] = items[smallest], items[i]
    return items


def merge_sort(items: list[int]) -> list[int]:
    """Sort a list with merge sort and return a new list."""
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left = merge_sort(items[:mid])
    right = merge_sort(items[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(items: list[int]) -> list[int]:
    """Sort a list with quicksort and return a new list."""
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)


def heap_sort(items: list[int]) -> list[int]:
    """Sort a list with a binary heap and return a new list."""
    heap = list(items)
    heapq.heapify(heap)
    return [heapq.heappop(heap) for _ in range(len(heap))]


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number.

    >>> fibonacci(10)
    55
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def factorial(n: int) -> int:
    """Return n factorial.

    >>> factorial(5)
    120
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of a and b."""
    return a * b // gcd(a, b)


def is_prime(n: int) -> bool:
    """Return True when n is a prime number."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Return every prime up to and including limit."""
    is_composite = [False] * (limit + 1)
    primes = []
    for number in range(2, limit + 1):
        if not is_composite[number]:
            primes.append(number)
            for multiple in range(number * number, limit + 1, number):
                is_composite[multiple] = True
    return primes


def is_palindrome(text: str) -> bool:
    """Return True when text reads the same backwards, ignoring case and non-letters."""
    cleaned = [c.lower() for c in text if c.isalnum()]
    return cleaned == cleaned[::-1]


def reverse_words(sentence: str) -> str:
    """Return the sentence with its words in reverse order."""
    return " ".join(reversed(sentence.split()))


def count_words(text: str) -> dict[str, int]:
    """Return how often each word occurs in text."""
    counts: dict[str, int] = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts


def two_sum(numbers: list[int], target: int) -> Optional[tuple[int, int]]:
    """Return the indices of two numbers that add up to target, or None."""
    seen: dict[int, int] = {}
    for index, number in enumerate(numbers):
        complement = target - number
        if complement in seen:
            return seen[complement], index
        seen[number] = index
    return None


def max_subarray(numbers: list[int]) -> int:
    """Return the largest sum of a contiguous subarray (Kadane's algorithm)."""
    best = current = numbers[0]
    for number in numbers[1:]:
        current = max(number, current + number)
        best = max(best, current)
    return best


def longest_common_subsequence(a: str, b: str) -> int:
    """Return the length of the longest common subsequence of a and b."""
    rows, cols = len(a) + 1, len(b) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def edit_distance(a: str, b: str) -> int:
    """Return the Levenshtein distance between a and b."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """Return the best total value of items that fit into capacity (0/1 knapsack)."""
    best = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for room in range(capacity, weight - 1, -1):
            best[room] = max(best[room], best[room - weight] + value)
    return best[capacity]


def coin_change(coins: list[int], amount: int) -> int:
    """Return the fewest coins that make up amount, or -1 when it cannot be made."""
    best = [0] + [amount + 1] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                best[value] = min(best[value], best[value - coin] + 1)
    return best[amount] if best[amount] <= amount else -1


def breadth_first_search(graph: dict[str, list[str]], start: str) -> list[str]:
    """Return the nodes of graph in breadth-first order from start."""
    visited = [start]
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return visited


def depth_first_search(graph: dict[str, list[str]], start: str) -> list[str]:
    """Return the nodes of graph in depth-first order from start."""
    visited: list[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            stack.extend(reversed(graph.get(node, [])))
    return visited


def dijkstra(graph: dict[str, dict[str, int]], source: str) -> dict[str, int]:
    """Return the shortest distance from source to every reachable node."""
    distances = {source: 0}
    queue = [(0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, float("inf")):
            continue
        for neighbour, weight in graph.get(node, {}).items():
            candidate = distance + weight
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distances


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Return the nodes of a directed acyclic graph in topological order (Kahn's algorithm)."""
    in_degree = {node: 0 for node in graph}
    for targets in graph.values():
        for target in targets:
            in_degree[target] = in_degree.get(target, 0) + 1
    queue = deque(node for node, degree in in_degree.items() if degree == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in graph.get(node, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    if len(order) != len(in_degree):
        raise ValueError("the graph has a cycle")
    return order


class UnionFind:
    """Disjoint sets with union by rank and path compression."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


class Stack:
    """A last-in, first-out stack."""

    def __init__(self) -> None:
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("peek at an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


class Queue:
    """A first-in, first-out queue."""

    def __init__(self) -> None:
        self._items: deque = deque()

    def enqueue(self, item) -> None:
        self._items.append(item)

    def dequeue(self):
        if not self._items:
            raise IndexError("dequeue from an empty queue")
        return self._items.popleft()

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


class Node:
    """A node of a singly linked list."""

    def __init__(self, value, next_node: Optional["Node"] = None) -> None:
        self.value = value
        self.next = next_node


class LinkedList:
    """A singly linked list."""

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.size = 0

    def append(self, value) -> None:
        node = Node(value)
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = node
        self.size += 1

    def prepend(self, value) -> None:
        self.head = Node(value, self.head)
        self.size += 1

    def reverse(self) -> None:
        previous = None
        current = self.head
        while current is not None:
            following = current.next
            current.next = previous
            previous = current
            current = following
        self.head = previous

    def to_list(self) -> list:
        values = []
        current = self.head
        while current is not None:
            values.append(current.value)
            current = current.next
        return values


class TreeNode:
    """A node of a binary search tree."""

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class BinarySearchTree:
    """An unbalanced binary search tree."""

    def __init__(self) -> None:
        self.root: Optional[TreeNode] = None

    def insert(self, key: int) -> None:
        if self.root is None:
            self.root = TreeNode(key)
            return
        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = TreeNode(key)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(key)
                    return
                current = current.right

    def contains(self, key: int) -> bool:
        current = self.root
        while current is not None:
            if key == current.key:
                return True
            current = current.left if key < current.key else current.right
        return False

    def in_order(self) -> list[int]:
        result: list[int] = []

        def walk(node: Optional[TreeNode]) -> None:
            if node is None:
                return
            walk(node.left)
            result.append(node.key)
            walk(node.right)

        walk(self.root)
        return result


class TrieNode:
    """A node of a prefix tree."""

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_word = False


class Trie:
    """A prefix tree of words."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class LRUCache:
    """A least-recently-used cache with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: dict = {}

    def get(self, key):
        if key not in self.items:
            return None
        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key, value) -> None:
        if key in self.items:
            self.items.pop(key)
        elif len(self.items) >= self.capacity:
            oldest = next(iter(self.items))
            del self.items[oldest]
        self.items[key] = value


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Return the matrix product a times b."""
    rows, inner, cols = len(a), len(b), len(b[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Return the transpose of a matrix."""
    return [list(row) for row in zip(*matrix)]


def flatten(nested: list) -> list:
    """Return a flat list of every value in a nested list."""
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def chunk(items: list, size: int) -> list[list]:
    """Split items into lists of at most size elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def caesar_cipher(text: str, shift: int) -> str:
    """Shift every letter of text by shift places in the alphabet."""
    result = []
    for char in text:
        if char.isalpha():
            base = ord("A") if char.isupper() else ord("a")
            result.append(chr((ord(char) - base + shift) % 26 + base))
        else:
            result.append(char)
    return "".join(result)


def run_length_encode(text: str) -> str:
    """Return the run-length encoding of text, e.g. "aaab" -> "3a1b"."""
    if not text:
        return ""
    encoded = []
    count = 1
    for previous, current in zip(text, text[1:]):
        if current == previous:
            count += 1
        else:
            encoded.append(f"{count}{previous}")
            count = 1
    encoded.append(f"{count}{text[-1]}")
    return "".join(encoded)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
