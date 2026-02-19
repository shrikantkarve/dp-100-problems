# 🧠 100 Dynamic Programming Problems

A comprehensive collection of **100 classic and advanced Dynamic Programming problems** implemented in Python, complete with detailed docstrings and unit tests.

---

## 📂 Structure

```
dp-tasks/
├── dp_solutions.py          # All 100 DP solutions
├── tests/
│   └── test_dp_solutions.py # Unit tests for every problem
└── README.md
```

---

## 🗂️ Problem Categories

| # | Category | Problems |
|---|----------|----------|
| 1–10 | **Classic 1D DP** | Fibonacci, House Robber, Coin Change, Word Break… |
| 11–20 | **Subsequence Problems** | LCS, LIS, Distinct Subsequences, Wiggle Subsequence… |
| 21–30 | **Knapsack Variants** | 0/1 Knapsack, Unbounded Knapsack, Subset Sum, Target Sum… |
| 31–40 | **Matrix / Grid DP** | Unique Paths, Min Path Sum, Maximal Square, Dungeon Game… |
| 41–50 | **String DP** | Edit Distance, Palindrome Partitioning, Regex Matching… |
| 51–60 | **Interval DP** | Burst Balloons, Matrix Chain Multiplication, Remove Boxes… |
| 61–70 | **Tree DP** | House Robber III, Tree Cameras, Max Path Sum, Tree Diameter… |
| 71–80 | **Bitmask DP** | Traveling Salesman, Min XOR Sum, Shortest Path All Nodes… |
| 81–90 | **Math / Counting DP** | Perfect Squares, Dice Rolls, Knight Probability, 21 Game… |
| 91–100 | **Advanced / Mixed** | Stock Trading, Job Scheduling, Tallest Billboard, Superstring… |

---

## 📋 Full Problem List

### Classic 1D DP (1–10)
1. **Fibonacci Number** – nth Fibonacci using O(1) space
2. **Climbing Stairs** – Count ways to reach top with 1 or 2 steps
3. **House Robber** – Max sum with no two adjacent
4. **Maximum Subarray** – Kadane's algorithm
5. **Min Cost Climbing Stairs** – Minimum cost to reach top
6. **Jump Game** – Can you reach the last index?
7. **Jump Game II** – Minimum jumps to last index
8. **Decode Ways** – Number of ways to decode a digit string
9. **Coin Change** – Minimum coins for a given amount
10. **Word Break** – Can string be segmented into dictionary words?

### Subsequence Problems (11–20)
11. **Longest Common Subsequence** – LCS length
12. **Longest Increasing Subsequence** – LIS length
13. **Longest Palindromic Subsequence** – Via LCS with reversed string
14. **Number of LIS** – Count all longest increasing subsequences
15. **Shortest Common Supersequence Length** – Via LCS
16. **Is Subsequence** – Check if s is subsequence of t
17. **Distinct Subsequences** – Count ways t appears in s as subsequence
18. **Maximum Length Pair Chain** – Greedy + DP chain
19. **Wiggle Subsequence** – Longest alternating subsequence
20. **Longest Arithmetic Subsequence** – Max length with constant difference

### Knapsack Variants (21–30)
21. **0/1 Knapsack** – Classic bounded knapsack
22. **Unbounded Knapsack** – Items can be reused
23. **Subset Sum** – Does any subset hit the target?
24. **Partition Equal Subset Sum** – Split into two equal halves
25. **Coin Change Ways** – Count combinations for amount
26. **Target Sum** – Ways to assign +/- to reach target
27. **Last Stone Weight II** – Minimum result after smashing
28. **Ones and Zeroes** – Largest subset under m zeros and n ones
29. **Profitable Schemes** – Count schemes with minimum profit
30. **Shopping Offers** – Min cost using bundle offers

### Matrix / Grid DP (31–40)
31. **Unique Paths** – Grid paths right/down only
32. **Unique Paths with Obstacles** – Blocked cells variant
33. **Minimum Path Sum** – Cheapest path in grid
34. **Triangle Min Path** – Top-to-bottom triangle traversal
35. **Maximal Square** – Largest all-1s square area
36. **Count Square Submatrices** – Total square submatrices with 1s
37. **Dungeon Game** – Min starting HP for dungeon traversal
38. **Cherry Pickup** – Max cherries on round trip
39. **Out of Boundary Paths** – Count paths leaving the grid
40. **Paths Divisible by K** – Count paths with sum divisible by k

### String DP (41–50)
41. **Edit Distance** – Levenshtein distance
42. **Longest Palindromic Substring Length** – DP table approach
43. **Min Insertions for Palindrome** – Fewest insertions
44. **Palindrome Partitioning Min Cuts** – Fewest cuts
45. **Scramble String** – Is one string a scramble of another?
46. **Interleaving String** – Is s3 an interleaving of s1 and s2?
47. **Regular Expression Matching** – `.` and `*` pattern match
48. **Wildcard Matching** – `?` and `*` wildcard match
49. **Count Palindromic Substrings** – Total palindrome substrings
50. **Longest Valid Parentheses** – Longest valid bracket substring

### Interval DP (51–60)
51. **Burst Balloons** – Max coins from optimal burst order
52. **Strange Printer** – Minimum print turns
53. **Min Cost to Merge Stones** – k-way merge minimum cost
54. **Remove Boxes** – Maximum points removing colored boxes
55. **Minimum Score Triangulation** – Cheapest polygon triangulation
56. **Matrix Chain Multiplication** – Minimum scalar multiplications
57. **Optimal BST Cost** – Binary search tree with minimum search cost
58. **Minimum Falling Path Sum** – Min sum falling through matrix
59. **Zuma Game** – Min balls to clear Zuma board
60. **Minimum Window Subsequence** – Shortest window containing t

### Tree DP (61–70)
61. **House Robber III** – Max rob in binary tree
62. **Diameter of Binary Tree** – Longest path between two nodes
63. **Binary Tree Cameras** – Minimum cameras to cover all nodes
64. **Maximum Path Sum** – Max sum path any node to any node
65. **Count Nodes in Complete Tree** – Efficient complete tree count
66. **Sum of Root to Leaf Numbers** – Sum all root-leaf numbers
67. **Longest Univalue Path** – Longest same-value adjacent path
68. **Find Duplicate Subtrees** – All duplicate subtree roots
69. **Max Product by Removing One Edge** – Best tree split
70. **Max Path Sum (any-to-any)** – Alias of problem 64

### Bitmask DP (71–80)
71. **Traveling Salesman Problem** – TSP exact solution
72. **Count Special Subsets** – Subsets where sum divides count
73. **Minimum XOR Sum** – Optimal assignment minimizing XOR
74. **Shortest Path Visiting All Nodes** – BFS + bitmask
75. **Count Ways to Assign Tasks** – Permutation counting
76. **Maximize Score After K Operations** – Greedy heap
77. **Minimum Incompatibility** – Min sum of max-min per group
78. **Distribute Repeating Integers** – Feasibility check
79. **Minimum Time to Finish Jobs** – Optimal job assignment
80. **Count Vowel Permutations** – Count n-length vowel strings

### Math / Counting DP (81–90)
81. **Perfect Squares** – Min squares summing to n
82. **Integer Break** – Max product of integer parts
83. **Count Numbers with Unique Digits** – Up to 10^n
84. **Number of Ways to Roll to Target** – Dice combinations
85. **Knight Probability** – Probability knight stays on board
86. **New 21 Game** – Probability score ≤ n
87. **Soup Servings** – Probability soup A empties first
88. **Ways to Make Change** – (Alias of problem 25)
89. **Count Stepping Numbers** – Numbers with adjacent digit diff = 1
90. **Count Digit DP** – Integers with digit sum divisible by 5

### Advanced / Mixed (91–100)
91. **Largest Divisible Subset** – Subset with pairwise divisibility
92. **Stock with Cooldown** – Max profit with cooldown period
93. **Stock with Transaction Fee** – Max profit with per-trade fee
94. **Stock with K Transactions** – Max profit, at most k buys/sells
95. **Minimum Difficulty Job Schedule** – Schedule over d days
96. **Paint Fence** – Ways to paint fence with k colors
97. **Number of Music Playlists** – Playlists of length n
98. **Count Palindromic Subsequences** – Distinct palindromic subsequences
99. **Shortest Superstring** – Shortest string containing all words
100. **Tallest Billboard** – Max height equal-support billboard

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pytest

```bash
pip install pytest
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/test_dp_solutions.py -v

# Run a specific category
python -m pytest tests/test_dp_solutions.py -k "Knapsack" -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=dp_solutions --cov-report=term-missing
```

### Using the Solutions

```python
from dp_solutions import (
    coin_change,
    longest_common_subsequence,
    burst_balloons,
)

print(coin_change([1, 5, 10, 25], 36))  # 3 coins: 25+10+1
print(longest_common_subsequence("abcde", "ace"))  # 3
print(burst_balloons([3, 1, 5, 8]))  # 167
```

---

## 🧩 Key DP Patterns

| Pattern | Problems |
|---------|----------|
| **1D Tabulation** | 1-10 |
| **2D DP Table** | 11, 17, 41, 46, 47 |
| **Rolling Array** | 21-25 (space optimized) |
| **Memoization** | 30, 45, 54, 59, 87 |
| **Interval [i,j]** | 51-57 |
| **Bitmask State** | 71-79 |
| **Tree Post-order** | 61-70 |
| **Digit DP** | 90 |
| **Profile DP** | 79, 95 |

---

## 📊 Complexity Overview

Most problems run in:
- **Time**: O(n²) to O(n³) for interval DP, O(2ⁿ · n) for bitmask DP
- **Space**: O(n) to O(n²), many optimized to O(n) or O(capacity)

---

## 🤝 Contributing

1. Fork the repo
2. Add your solution with a clear docstring
3. Add unit tests in `tests/test_dp_solutions.py`
4. Submit a pull request

---

## 📄 License

MIT License — free to use, modify, and distribute.
