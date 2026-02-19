"""
100 Dynamic Programming Problems - Solutions
=============================================
Organized by category:
  1-10:   Classic 1D DP
  11-20:  Subsequence Problems
  21-30:  Knapsack Variants
  31-40:  Matrix/Grid DP
  41-50:  String DP
  51-60:  Interval DP
  61-70:  Tree DP
  71-80:  Bitmask DP
  81-90:  Math/Counting DP
  91-100: Advanced / Mixed
"""

from functools import lru_cache
from typing import List, Optional
import math


# ─────────────────────────────────────────────
# 1-10: CLASSIC 1D DP
# ─────────────────────────────────────────────

def fibonacci(n: int) -> int:
    """1. nth Fibonacci number (bottom-up)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def climbing_stairs(n: int) -> int:
    """2. Number of ways to climb n stairs (1 or 2 steps at a time)."""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def house_robber(nums: List[int]) -> int:
    """3. Maximum money robbed from houses (no two adjacent)."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = 0, 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + n)
    return prev1


def max_subarray(nums: List[int]) -> int:
    """4. Maximum subarray sum (Kadane's algorithm)."""
    max_sum = cur = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        max_sum = max(max_sum, cur)
    return max_sum


def min_cost_climbing_stairs(cost: List[int]) -> int:
    """5. Minimum cost to reach the top of the floor."""
    n = len(cost)
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]


def jump_game(nums: List[int]) -> bool:
    """6. Can you reach the last index?"""
    max_reach = 0
    for i, n in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + n)
    return True


def jump_game_ii(nums: List[int]) -> int:
    """7. Minimum jumps to reach the last index."""
    jumps = cur_end = cur_far = 0
    for i in range(len(nums) - 1):
        cur_far = max(cur_far, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = cur_far
    return jumps


def decode_ways(s: str) -> int:
    """8. Number of ways to decode a digit string."""
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    return dp[n]


def coin_change(coins: List[int], amount: int) -> int:
    """9. Minimum number of coins to make up amount."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


def word_break(s: str, word_dict: List[str]) -> bool:
    """10. Can the string be segmented into dictionary words?"""
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]


# ─────────────────────────────────────────────
# 11-20: SUBSEQUENCE PROBLEMS
# ─────────────────────────────────────────────

def longest_common_subsequence(text1: str, text2: str) -> int:
    """11. Length of longest common subsequence."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def longest_increasing_subsequence(nums: List[int]) -> int:
    """12. Length of longest strictly increasing subsequence."""
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def longest_palindromic_subsequence(s: str) -> int:
    """13. Length of longest palindromic subsequence."""
    return longest_common_subsequence(s, s[::-1])


def number_of_lis(nums: List[int]) -> int:
    """14. Number of longest increasing subsequences."""
    n = len(nums)
    if n == 0:
        return 0
    lengths = [1] * n
    counts = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
    max_len = max(lengths)
    return sum(c for l, c in zip(lengths, counts) if l == max_len)


def shortest_common_supersequence_len(s1: str, s2: str) -> int:
    """15. Length of shortest common supersequence."""
    lcs = longest_common_subsequence(s1, s2)
    return len(s1) + len(s2) - lcs


def is_subsequence(s: str, t: str) -> bool:
    """16. Is s a subsequence of t?"""
    i = 0
    for c in t:
        if i < len(s) and c == s[i]:
            i += 1
    return i == len(s)


def distinct_subsequences(s: str, t: str) -> int:
    """17. Number of distinct subsequences of s that equal t."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]
    return dp[m][n]


def max_length_pair_chain(pairs: List[List[int]]) -> int:
    """18. Maximum length of chain of pairs."""
    pairs.sort(key=lambda x: x[1])
    count, cur_end = 0, float('-inf')
    for a, b in pairs:
        if a > cur_end:
            count += 1
            cur_end = b
    return count


def wiggle_subsequence(nums: List[int]) -> int:
    """19. Length of longest wiggle subsequence."""
    up = down = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            up = down + 1
        elif nums[i] < nums[i-1]:
            down = up + 1
    return max(up, down)


def longest_arithmetic_subsequence(nums: List[int]) -> int:
    """20. Length of longest arithmetic subsequence."""
    dp = [{} for _ in range(len(nums))]
    res = 2
    for i in range(1, len(nums)):
        for j in range(i):
            diff = nums[i] - nums[j]
            dp[i][diff] = dp[j].get(diff, 1) + 1
            res = max(res, dp[i][diff])
    return res


# ─────────────────────────────────────────────
# 21-30: KNAPSACK VARIANTS
# ─────────────────────────────────────────────

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """21. 0/1 Knapsack — max value within capacity."""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """22. Unbounded Knapsack — items can be reused."""
    dp = [0] * (capacity + 1)
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


def subset_sum(nums: List[int], target: int) -> bool:
    """23. Does any subset sum to target?"""
    dp = {0}
    for n in nums:
        dp |= {x + n for x in dp}
    return target in dp


def partition_equal_subset_sum(nums: List[int]) -> bool:
    """24. Can the array be partitioned into two equal-sum subsets?"""
    total = sum(nums)
    if total % 2:
        return False
    return subset_sum(nums, total // 2)


def coin_change_ways(coins: List[int], amount: int) -> int:
    """25. Number of combinations that make up amount."""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]


def target_sum(nums: List[int], target: int) -> int:
    """26. Number of ways to assign +/- to reach target."""
    dp = {0: 1}
    for n in nums:
        new_dp = {}
        for s, cnt in dp.items():
            new_dp[s + n] = new_dp.get(s + n, 0) + cnt
            new_dp[s - n] = new_dp.get(s - n, 0) + cnt
        dp = new_dp
    return dp.get(target, 0)


def last_stone_weight_ii(stones: List[int]) -> int:
    """27. Minimum possible result of smashing stones."""
    total = sum(stones)
    target = total // 2
    dp = {0}
    for s in stones:
        dp = {x + s for x in dp} | dp
    return min(total - 2 * x for x in dp if x <= target)


def ones_and_zeroes(strs: List[str], m: int, n: int) -> int:
    """28. Largest subset with at most m 0s and n 1s."""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones] + 1)
    return dp[m][n]


def profitable_schemes(n: int, min_profit: int, group: List[int], profit: List[int]) -> int:
    """29. Number of schemes with at least min_profit using at most n members."""
    MOD = 10**9 + 7
    dp = [[0] * (min_profit + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for g, p in zip(group, profit):
        for j in range(n, g - 1, -1):
            for k in range(min_profit, -1, -1):
                dp[j][min(min_profit, k + p)] = (dp[j][min(min_profit, k + p)] + dp[j-g][k]) % MOD
    return sum(dp[j][min_profit] for j in range(n + 1)) % MOD


def shopping_offers(price: List[int], special: List[List[int]], needs: List[int]) -> int:
    """30. Minimum cost to satisfy needs with special offers."""
    @lru_cache(maxsize=None)
    def dp(needs_tuple):
        needs = list(needs_tuple)
        cost = sum(n * p for n, p in zip(needs, price))
        for offer in special:
            new_needs = [needs[i] - offer[i] for i in range(len(needs))]
            if all(x >= 0 for x in new_needs):
                cost = min(cost, offer[-1] + dp(tuple(new_needs)))
        return cost
    return dp(tuple(needs))


# ─────────────────────────────────────────────
# 31-40: MATRIX / GRID DP
# ─────────────────────────────────────────────

def unique_paths(m: int, n: int) -> int:
    """31. Unique paths in m×n grid (only right/down)."""
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[-1]


def unique_paths_with_obstacles(grid: List[List[int]]) -> int:
    """32. Unique paths with obstacles (0=free, 1=blocked)."""
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = 1 if grid[0][0] == 0 else 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[-1]


def min_path_sum(grid: List[List[int]]) -> int:
    """33. Minimum path sum in a grid."""
    m, n = len(grid), len(grid[0])
    dp = grid[0][:]
    for j in range(1, n):
        dp[j] += dp[j-1]
    for i in range(1, m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])
    return dp[-1]


def triangle_min_path(triangle: List[List[int]]) -> int:
    """34. Minimum path sum from top to bottom of triangle."""
    dp = triangle[-1][:]
    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
    return dp[0]


def maximal_square(matrix: List[List[str]]) -> int:
    """35. Area of largest square containing only 1s."""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side


def count_square_submatrices(matrix: List[List[int]]) -> int:
    """36. Count square submatrices with all 1s."""
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] and i > 0 and j > 0:
                matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1
            total += matrix[i][j]
    return total


def dungeon_game(dungeon: List[List[int]]) -> int:
    """37. Minimum initial health to reach bottom-right."""
    m, n = len(dungeon), len(dungeon[0])
    dp = [[0] * n for _ in range(m)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                dp[i][j] = max(1, 1 - dungeon[i][j])
            elif i == m - 1:
                dp[i][j] = max(1, dp[i][j+1] - dungeon[i][j])
            elif j == n - 1:
                dp[i][j] = max(1, dp[i+1][j] - dungeon[i][j])
            else:
                dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
    return dp[0][0]


def cherry_pickup(grid: List[List[int]]) -> int:
    """38. Maximum cherries collected going and coming back."""
    n = len(grid)
    memo = {}

    def dp(r1, c1, r2):
        c2 = r1 + c1 - r2
        if r1 >= n or c1 >= n or r2 >= n or c2 >= n:
            return float('-inf')
        if grid[r1][c1] == -1 or grid[r2][c2] == -1:
            return float('-inf')
        if r1 == n - 1 and c1 == n - 1:
            return grid[r1][c1]
        if (r1, c1, r2) in memo:
            return memo[(r1, c1, r2)]
        cherries = grid[r1][c1] + (0 if r1 == r2 else grid[r2][c2])
        best = max(dp(r1+1,c1,r2+1), dp(r1,c1+1,r2), dp(r1+1,c1,r2), dp(r1,c1+1,r2+1))
        res = cherries + best
        memo[(r1, c1, r2)] = res
        return res

    return max(0, dp(0, 0, 0))


def out_of_boundary_paths(m: int, n: int, max_move: int, start_row: int, start_col: int) -> int:
    """39. Number of paths to move ball out of boundary."""
    MOD = 10**9 + 7
    dp = [[0] * n for _ in range(m)]
    dp[start_row][start_col] = 1
    count = 0
    for _ in range(max_move):
        new_dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        new_dp[ni][nj] = (new_dp[ni][nj] + dp[i][j]) % MOD
                    else:
                        count = (count + dp[i][j]) % MOD
        dp = new_dp
    return count


def number_of_paths_divisible(grid: List[List[int]], k: int) -> int:
    """40. Number of paths where sum is divisible by k."""
    MOD = 10**9 + 7
    m, n = len(grid), len(grid[0])
    dp = [[0] * k for _ in range(n)]
    dp[0][grid[0][0] % k] = 1
    for j in range(1, n):
        r = (grid[0][j]) % k
        for rem in range(k):
            dp[j][(rem + r) % k] = dp[j-1][rem]
    for i in range(1, m):
        new_dp = [[0] * k for _ in range(n)]
        r = grid[i][0] % k
        for rem in range(k):
            new_dp[0][(rem + r) % k] = dp[0][rem]
        for j in range(1, n):
            r = grid[i][j] % k
            for rem in range(k):
                new_dp[j][(rem + r) % k] = (new_dp[j][(rem + r) % k] + dp[j][rem] + new_dp[j-1][rem]) % MOD
        dp = new_dp
    return dp[n-1][0]


# ─────────────────────────────────────────────
# 41-50: STRING DP
# ─────────────────────────────────────────────

def edit_distance(word1: str, word2: str) -> int:
    """41. Minimum edit distance (Levenshtein)."""
    m, n = len(word1), len(word2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if word1[i-1] == word2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def longest_palindromic_substring_len(s: str) -> int:
    """42. Length of longest palindromic substring."""
    n = len(s)
    if n == 0:
        return 0
    dp = [[False] * n for _ in range(n)]
    max_len = 1
    for i in range(n):
        dp[i][i] = True
    for i in range(n - 1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            max_len = 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                max_len = max(max_len, length)
    return max_len


def min_insertions_palindrome(s: str) -> int:
    """43. Minimum insertions to make string palindrome."""
    n = len(s)
    lps = longest_common_subsequence(s, s[::-1])
    return n - lps


def palindrome_partitioning_min_cuts(s: str) -> int:
    """44. Minimum cuts to partition string into palindromes."""
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for i in range(n - 1):
        is_pal[i][i+1] = s[i] == s[i+1]
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = s[i] == s[j] and is_pal[i+1][j-1]
    dp = list(range(-1, n))
    for i in range(n):
        if is_pal[0][i]:
            dp[i+1] = 0
        else:
            for j in range(i):
                if is_pal[j+1][i]:
                    dp[i+1] = min(dp[i+1], dp[j+1] + 1)
    return dp[n]


def scramble_string(s1: str, s2: str) -> bool:
    """45. Is s2 a scramble of s1?"""
    @lru_cache(maxsize=None)
    def dp(a, b):
        if a == b:
            return True
        if sorted(a) != sorted(b):
            return False
        n = len(a)
        for i in range(1, n):
            if (dp(a[:i], b[:i]) and dp(a[i:], b[i:])) or \
               (dp(a[:i], b[n-i:]) and dp(a[i:], b[:n-i])):
                return True
        return False
    return dp(s1, s2)


def interleaving_string(s1: str, s2: str, s3: str) -> bool:
    """46. Is s3 formed by interleaving s1 and s2?"""
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    return dp[m][n]


def regular_expression_matching(s: str, p: str) -> bool:
    """47. Regular expression matching with '.' and '*'."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or (dp[i-1][j] and (p[j-2] == s[i-1] or p[j-2] == '.'))
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]


def wildcard_matching(s: str, p: str) -> bool:
    """48. Wildcard pattern matching with '?' and '*'."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]


def count_palindromic_substrings(s: str) -> int:
    """49. Count palindromic substrings."""
    n, count = len(s), 0
    for center in range(2 * n - 1):
        l, r = center // 2, (center + 1) // 2
        while l >= 0 and r < n and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    return count


def longest_valid_parentheses(s: str) -> int:
    """50. Length of longest valid parentheses substring."""
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len


# ─────────────────────────────────────────────
# 51-60: INTERVAL DP
# ─────────────────────────────────────────────

def burst_balloons(nums: List[int]) -> int:
    """51. Maximum coins by bursting balloons."""
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(dp[left][right],
                    nums[left] * nums[k] * nums[right] + dp[left][k] + dp[k][right])
    return dp[0][n-1]


def strange_printer(s: str) -> int:
    """52. Minimum turns for a strange printer to print string."""
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            dp[i][j] = dp[i][j-1] + 1
            for k in range(i, j):
                if s[k] == s[j]:
                    dp[i][j] = min(dp[i][j], (dp[i][k] if k > i else 0) + (dp[k+1][j-1] if k+1 <= j-1 else 0) + 1 if k+1<=j-1 else (dp[i][k] if k>i else 0) + 1)
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            dp[i][j] = dp[i][j-1] + 1
            for k in range(i, j):
                if s[k] == s[j]:
                    val = dp[i+1][k] if i+1 <= k else 0
                    dp[i][j] = min(dp[i][j], val + dp[k][j-1] if k <= j-1 else val)
    # Simpler re-implementation
    memo = {}
    def solve(i, j):
        if i > j: return 0
        if i == j: return 1
        if (i, j) in memo: return memo[(i, j)]
        res = solve(i, j - 1) + 1
        for k in range(i, j):
            if s[k] == s[j]:
                res = min(res, solve(i, k) + solve(k + 1, j - 1))
        memo[(i, j)] = res
        return res
    return solve(0, n - 1)


def minimum_cost_to_merge_stones(stones: List[int], k: int) -> int:
    """53. Minimum cost to merge piles of stones."""
    n = len(stones)
    if (n - 1) % (k - 1) != 0:
        return -1
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + stones[i]
    dp = [[0] * n for _ in range(n)]
    for length in range(k, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for mid in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid+1][j])
            if (j - i) % (k - 1) == 0:
                dp[i][j] += prefix[j+1] - prefix[i]
    return dp[0][n-1]


def remove_boxes(boxes: List[int]) -> int:
    """54. Maximum points from removing boxes."""
    @lru_cache(maxsize=None)
    def dp(l, r, k):
        if l > r:
            return 0
        while l < r and boxes[r] == boxes[r-1]:
            r -= 1
            k += 1
        res = dp(l, r-1, 0) + (k+1)**2
        for i in range(l, r):
            if boxes[i] == boxes[r]:
                res = max(res, dp(i+1, r-1, 0) + dp(l, i, k+1))
        return res
    return dp(0, len(boxes)-1, 0)


def minimum_score_triangulation(values: List[int]) -> int:
    """55. Minimum score to triangulate a polygon."""
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            dp[i][j] = float('inf')
            for k in range(i+1, j):
                dp[i][j] = min(dp[i][j], values[i]*values[j]*values[k] + dp[i][k] + dp[k][j])
    return dp[0][n-1]


def matrix_chain_multiplication(dims: List[int]) -> int:
    """56. Minimum multiplications for matrix chain."""
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n-1]


def optimal_bst_cost(freq: List[int]) -> int:
    """57. Optimal binary search tree minimum cost."""
    n = len(freq)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + freq[i]
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = freq[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            freq_sum = prefix[j+1] - prefix[i]
            for k in range(i, j + 1):
                left = dp[i][k-1] if k > i else 0
                right = dp[k+1][j] if k < j else 0
                dp[i][j] = min(dp[i][j], left + right + freq_sum)
    return dp[0][n-1]


def minimum_falling_path_sum(matrix: List[List[int]]) -> int:
    """58. Minimum falling path sum."""
    dp = matrix[0][:]
    for i in range(1, len(matrix)):
        new_dp = []
        for j in range(len(matrix[0])):
            best = dp[j]
            if j > 0: best = min(best, dp[j-1])
            if j < len(matrix[0])-1: best = min(best, dp[j+1])
            new_dp.append(matrix[i][j] + best)
        dp = new_dp
    return min(dp)


def zuma_game(board: str) -> int:
    """59. Minimum balls to clear the Zuma board."""
    @lru_cache(maxsize=None)
    def dp(s):
        if not s: return 0
        res = float('inf')
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            need = max(0, 3 - (j - i))
            res = min(res, need + dp(s[:i] + s[j:]))
            i = j
        return res
    return dp(board)


def minimum_window_subsequence_len(s: str, t: str) -> int:
    """60. Minimum length window in s containing t as subsequence."""
    min_len = float('inf')
    i = 0
    while i < len(s):
        j = 0
        start = i
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                j += 1
            i += 1
        if j == len(t):
            end = i
            i -= 1
            while j > 0:
                i -= 1
                if s[i] == t[j-1]:
                    j -= 1
            min_len = min(min_len, end - i - 1)
            i += 1
    return min_len if min_len != float('inf') else -1


# ─────────────────────────────────────────────
# 61-70: TREE DP
# ─────────────────────────────────────────────

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rob_houses_iii(root: Optional[TreeNode]) -> int:
    """61. Maximum amount robbed from a binary tree."""
    def dp(node):
        if not node:
            return 0, 0
        left_rob, left_skip = dp(node.left)
        right_rob, right_skip = dp(node.right)
        rob = node.val + left_skip + right_skip
        skip = max(left_rob, left_skip) + max(right_rob, right_skip)
        return rob, skip
    return max(dp(root))


def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """62. Diameter of a binary tree."""
    max_diameter = [0]
    def height(node):
        if not node:
            return 0
        l, r = height(node.left), height(node.right)
        max_diameter[0] = max(max_diameter[0], l + r)
        return 1 + max(l, r)
    height(root)
    return max_diameter[0]


def binary_tree_cameras(root: Optional[TreeNode]) -> int:
    """63. Minimum cameras to monitor all nodes."""
    cameras = [0]
    def dp(node):
        if not node:
            return 2  # covered
        left, right = dp(node.left), dp(node.right)
        if left == 0 or right == 0:
            cameras[0] += 1
            return 1
        if left == 1 or right == 1:
            return 2
        return 0
    if dp(root) == 0:
        cameras[0] += 1
    return cameras[0]


def max_path_sum(root: Optional[TreeNode]) -> int:
    """64. Maximum path sum in a binary tree."""
    max_sum = [float('-inf')]
    def dp(node):
        if not node:
            return 0
        left = max(dp(node.left), 0)
        right = max(dp(node.right), 0)
        max_sum[0] = max(max_sum[0], node.val + left + right)
        return node.val + max(left, right)
    dp(root)
    return max_sum[0]


def count_nodes_complete_tree(root: Optional[TreeNode]) -> int:
    """65. Count nodes in a complete binary tree efficiently."""
    if not root:
        return 0
    left_h = right_h = 0
    l, r = root, root
    while l:
        left_h += 1
        l = l.left
    while r:
        right_h += 1
        r = r.right
    if left_h == right_h:
        return (1 << left_h) - 1
    return 1 + count_nodes_complete_tree(root.left) + count_nodes_complete_tree(root.right)


def sum_root_to_leaf_numbers(root: Optional[TreeNode]) -> int:
    """66. Sum of all root-to-leaf numbers."""
    def dp(node, cur):
        if not node:
            return 0
        cur = cur * 10 + node.val
        if not node.left and not node.right:
            return cur
        return dp(node.left, cur) + dp(node.right, cur)
    return dp(root, 0)


def longest_univalue_path(root: Optional[TreeNode]) -> int:
    """67. Longest path where each pair of adjacent nodes has same value."""
    res = [0]
    def dp(node):
        if not node:
            return 0
        l = dp(node.left) + 1 if node.left and node.left.val == node.val else 0
        r = dp(node.right) + 1 if node.right and node.right.val == node.val else 0
        res[0] = max(res[0], l + r)
        return max(l, r)
    dp(root)
    return res[0]


def find_duplicate_subtrees(root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
    """68. Find all duplicate subtrees."""
    from collections import defaultdict
    count = defaultdict(int)
    res = []
    def serialize(node):
        if not node:
            return '#'
        s = f"{node.val},{serialize(node.left)},{serialize(node.right)}"
        count[s] += 1
        if count[s] == 2:
            res.append(node)
        return s
    serialize(root)
    return res


def max_product_subtree(root: Optional[TreeNode]) -> int:
    """69. Maximum product of two subtrees split by removing one edge."""
    MOD = 10**9 + 7
    total_sum = [0]
    def total(node):
        if not node:
            return 0
        s = node.val + total(node.left) + total(node.right)
        total_sum[0] = max(total_sum[0], s)
        return s
    total_sum[0] = total(root)
    T = total_sum[0]
    best = [0]
    def dp(node):
        if not node:
            return 0
        s = node.val + dp(node.left) + dp(node.right)
        best[0] = max(best[0], s * (T - s))
        return s
    dp(root)
    return best[0] % MOD


def binary_tree_max_sum_path_any_to_any(root: Optional[TreeNode]) -> int:
    """70. Alias for problem 64 — max path sum any node to any node."""
    return max_path_sum(root)


# ─────────────────────────────────────────────
# 71-80: BITMASK DP
# ─────────────────────────────────────────────

def traveling_salesman(dist: List[List[int]]) -> int:
    """71. Traveling Salesman Problem (TSP) via bitmask DP."""
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u & 1) or dp[mask][u] == INF:
                continue
            for v in range(n):
                if mask >> v & 1:
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
    full = (1 << n) - 1
    return min(dp[full][u] + dist[u][0] for u in range(n))


def count_special_subsets(nums: List[int]) -> int:
    """72. Count subsets using bitmask enumeration."""
    n = len(nums)
    count = 0
    for mask in range(1, 1 << n):
        subset = [nums[i] for i in range(n) if mask >> i & 1]
        if sum(subset) % len(subset) == 0:
            count += 1
    return count


def minimum_xor_sum(nums1: List[int], nums2: List[int]) -> int:
    """73. Minimum XOR sum of two arrays (assignment problem)."""
    n = len(nums1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        i = bin(mask).count('1')
        if i >= n:
            continue
        for j in range(n):
            if not (mask >> j & 1):
                dp[mask | (1 << j)] = min(dp[mask | (1 << j)], dp[mask] + (nums1[i] ^ nums2[j]))
    return dp[(1 << n) - 1]


def shortest_path_visiting_all_nodes(graph: List[List[int]]) -> int:
    """74. Shortest path to visit all nodes in a graph."""
    from collections import deque
    n = len(graph)
    full = (1 << n) - 1
    queue = deque()
    visited = set()
    for i in range(n):
        state = (i, 1 << i)
        queue.append((state, 0))
        visited.add(state)
    while queue:
        (node, mask), dist = queue.popleft()
        if mask == full:
            return dist
        for nei in graph[node]:
            new_mask = mask | (1 << nei)
            state = (nei, new_mask)
            if state not in visited:
                visited.add(state)
                queue.append((state, dist + 1))
    return -1


def count_ways_to_assign_tasks(n: int, tasks: int) -> int:
    """75. Number of ways to assign tasks (bitmask count)."""
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        bits = bin(mask).count('1')
        if bits >= tasks:
            continue
        for i in range(n):
            if not (mask >> i & 1):
                dp[mask | (1 << i)] += dp[mask]
    return dp[(1 << n) - 1]


def maximize_score_after_k_ops(nums: List[int], k: int) -> int:
    """76. Maximize score: pick element, score += nums[i], replace with ceil(nums[i]/3)."""
    import heapq
    heap = [-n for n in nums]
    heapq.heapify(heap)
    score = 0
    for _ in range(k):
        val = -heapq.heappop(heap)
        score += val
        heapq.heappush(heap, -math.ceil(val / 3))
    return score


def minimum_incompatibility(nums: List[int], k: int) -> int:
    """77. Minimum incompatibility of k subsets."""
    n = len(nums)
    size = n // k
    INF = float('inf')
    cost = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == size:
            elems = [nums[i] for i in range(n) if mask >> i & 1]
            if len(elems) == len(set(elems)):
                cost[mask] = max(elems) - min(elems)
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        remaining = ((1 << n) - 1) ^ mask
        sub = remaining
        while sub:
            if sub in cost:
                dp[mask | sub] = min(dp[mask | sub], dp[mask] + cost[sub])
            sub = (sub - 1) & remaining
    return dp[(1 << n) - 1] if dp[(1 << n) - 1] != INF else -1


def distribute_repeating_integers(quantity: List[int], nums_count: List[int]) -> bool:
    """78. Can quantities be satisfied by groups of identical integers?"""
    m = len(quantity)
    psum = [0] * (1 << m)
    for mask in range(1, 1 << m):
        for i in range(m):
            if mask >> i & 1:
                psum[mask] = psum[mask ^ (1 << i)] + quantity[i]
                break
    dp = [False] * (1 << m)
    dp[0] = True
    for cnt in nums_count:
        new_dp = dp[:]
        for mask in range((1 << m) - 1, -1, -1):
            if not dp[mask]:
                continue
            sub = ((1 << m) - 1) ^ mask
            while sub:
                if psum[sub] <= cnt:
                    new_dp[mask | sub] = True
                sub = (sub - 1) & (((1 << m) - 1) ^ mask)
        dp = new_dp
    return dp[(1 << m) - 1]


def find_minimum_time_to_finish_jobs(jobs: List[int], k: int) -> int:
    """79. Minimum maximum working time to assign jobs to k workers."""
    n = len(jobs)
    sub_sum = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if mask >> i & 1:
                sub_sum[mask] = sub_sum[mask ^ (1 << i)] + jobs[i]
                break
    INF = float('inf')
    dp = [[INF] * (1 << n) for _ in range(k + 1)]
    dp[0][0] = 0
    for i in range(1, k + 1):
        for mask in range(1 << n):
            sub = mask
            while sub:
                if dp[i-1][mask ^ sub] != INF:
                    dp[i][mask] = min(dp[i][mask], max(dp[i-1][mask ^ sub], sub_sum[sub]))
                sub = (sub - 1) & mask
    return dp[k][(1 << n) - 1]


def count_vowels_permutations(n: int) -> int:
    """80. Count strings of length n using vowel rules."""
    MOD = 10**9 + 7
    a = e = i = o = u = 1
    for _ in range(n - 1):
        a, e, i, o, u = (e + i + u), (a + i), (e + o), (i,), (o + i)
        a, e, i, o, u = a % MOD, e % MOD, i[0] % MOD, o % MOD, u % MOD
    return (a + e + i + o + u) % MOD


# ─────────────────────────────────────────────
# 81-90: MATH / COUNTING DP
# ─────────────────────────────────────────────

def perfect_squares(n: int) -> int:
    """81. Minimum number of perfect squares that sum to n."""
    dp = list(range(n + 1))
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j*j] + 1)
            j += 1
    return dp[n]


def integer_break(n: int) -> int:
    """82. Maximum product of parts that sum to n."""
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * max(i - j, dp[i - j]))
    return dp[n]


def count_numbers_with_unique_digits(n: int) -> int:
    """83. Count numbers with unique digits in range [0, 10^n]."""
    if n == 0:
        return 1
    res, unique = 10, 9
    available = 9
    for i in range(2, min(n, 10) + 1):
        unique *= available
        res += unique
        available -= 1
    return res


def num_rolls_to_target(n: int, k: int, target: int) -> int:
    """84. Number of ways to roll n dice with k faces to reach target."""
    MOD = 10**9 + 7
    dp = {0: 1}
    for _ in range(n):
        new_dp = {}
        for val, cnt in dp.items():
            for face in range(1, k + 1):
                new_dp[val + face] = (new_dp.get(val + face, 0) + cnt) % MOD
        dp = new_dp
    return dp.get(target, 0)


def knight_probability(n: int, k: int, row: int, col: int) -> float:
    """85. Probability knight stays on board after k moves."""
    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    dp = [[0.0] * n for _ in range(n)]
    dp[row][col] = 1.0
    for _ in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if dp[r][c]:
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            new_dp[nr][nc] += dp[r][c] / 8
        dp = new_dp
    return sum(sum(row) for row in dp)


def new_21_game(n: int, k: int, max_pts: int) -> float:
    """86. Probability score <= n when drawing cards up to k."""
    if k == 0 or n >= k + max_pts:
        return 1.0
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    win_sum = 1.0
    for i in range(1, n + 1):
        dp[i] = win_sum / max_pts
        if i < k:
            win_sum += dp[i]
        if i >= max_pts:
            win_sum -= dp[i - max_pts]
    return sum(dp[k:])


def soup_servings(n: int) -> float:
    """87. Probability soup A empties first or simultaneously."""
    if n > 4800:
        return 1.0
    m = math.ceil(n / 25)
    @lru_cache(maxsize=None)
    def dp(a, b):
        if a <= 0 and b <= 0: return 0.5
        if a <= 0: return 1.0
        if b <= 0: return 0.0
        return 0.25 * (dp(a-4,b) + dp(a-3,b-1) + dp(a-2,b-2) + dp(a-1,b-3))
    return dp(m, m)


def ways_to_make_change(amount: int, coins: List[int]) -> int:
    """88. Alias for coin_change_ways — number of combinations."""
    return coin_change_ways(coins, amount)


def count_stepping_numbers(low: str, high: str) -> List[int]:
    """89. Count stepping numbers in range (digit difference = 1)."""
    def get_stepping(n_digits):
        results = []
        if n_digits == 1:
            return list(range(10))
        prev = [[d] for d in range(1, 10)]
        for _ in range(n_digits - 1):
            new = []
            for seq in prev:
                last = seq[-1]
                if last > 0:
                    new.append(seq + [last - 1])
                if last < 9:
                    new.append(seq + [last + 1])
            prev = new
        for seq in prev:
            results.append(int(''.join(map(str, seq))))
        return results

    results = []
    low_int, high_int = int(low), int(high)
    max_digits = len(high)
    seen = set()
    for d in range(1, max_digits + 1):
        for num in get_stepping(d):
            if low_int <= num <= high_int and num not in seen:
                seen.add(num)
                results.append(num)
    return sorted(results)


def count_digit_dp(n: int) -> int:
    """90. Count integers from 1 to n where digit sum is divisible by 5."""
    s = str(n)
    length = len(s)
    @lru_cache(maxsize=None)
    def dp(pos, remainder, tight, started):
        if pos == length:
            return 1 if (not started or remainder == 0) else 0
        limit = int(s[pos]) if tight else 9
        result = 0
        for digit in range(0, limit + 1):
            new_remainder = (remainder + digit) % 5 if (started or digit != 0) else 0
            new_started = started or digit != 0
            result += dp(pos + 1, new_remainder, tight and digit == limit, new_started)
        return result
    return dp(0, 0, True, False)


# ─────────────────────────────────────────────
# 91-100: ADVANCED / MIXED
# ─────────────────────────────────────────────

def largest_divisible_subset(nums: List[int]) -> List[int]:
    """91. Largest subset where every pair is divisible."""
    if not nums:
        return []
    nums.sort()
    n = len(nums)
    dp = [1] * n
    parent = list(range(n))
    for i in range(1, n):
        for j in range(i):
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    max_idx = dp.index(max(dp))
    result = []
    while parent[max_idx] != max_idx:
        result.append(nums[max_idx])
        max_idx = parent[max_idx]
    result.append(nums[max_idx])
    return result[::-1]


def best_time_to_buy_sell_stock_with_cooldown(prices: List[int]) -> int:
    """92. Max profit with cooldown after selling."""
    hold = float('-inf')
    sold = 0
    rest = 0
    for p in prices:
        prev_hold = hold
        hold = max(hold, rest - p)
        rest = max(rest, sold)
        sold = prev_hold + p
    return max(sold, rest)


def best_time_with_transaction_fee(prices: List[int], fee: int) -> int:
    """93. Max profit with transaction fee."""
    hold = float('-inf')
    cash = 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p - fee)
    return cash


def stock_k_transactions(k: int, prices: List[int]) -> int:
    """94. Max profit with at most k transactions."""
    n = len(prices)
    if not n or k == 0:
        return 0
    if k >= n // 2:
        return sum(max(prices[i+1]-prices[i],0) for i in range(n-1))
    buy = [float('-inf')] * k
    sell = [0] * k
    for p in prices:
        for i in range(k):
            buy[i] = max(buy[i], (sell[i-1] if i > 0 else 0) - p)
            sell[i] = max(sell[i], buy[i] + p)
    return sell[-1]


def minimum_difficulty_job_schedule(job_difficulty: List[int], d: int) -> int:
    """95. Minimum difficulty to schedule jobs over d days."""
    n = len(job_difficulty)
    if n < d:
        return -1
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(d + 1)]
    dp[0][0] = 0
    for day in range(1, d + 1):
        for i in range(day, n - d + day + 1):
            max_diff = 0
            for j in range(i, day - 1, -1):
                max_diff = max(max_diff, job_difficulty[j-1])
                if dp[day-1][j-1] < INF:
                    dp[day][i] = min(dp[day][i], dp[day-1][j-1] + max_diff)
    return dp[d][n] if dp[d][n] < INF else -1


def paint_fence(n: int, k: int) -> int:
    """96. Number of ways to paint n posts with k colors."""
    if n == 0 or k == 0:
        return 0
    if n == 1:
        return k
    same = k
    diff = k * (k - 1)
    for _ in range(2, n):
        same, diff = diff, (same + diff) * (k - 1)
    return same + diff


def number_of_music_playlists(n: int, goal: int, k: int) -> int:
    """97. Number of playlists of length goal using n songs."""
    MOD = 10**9 + 7
    dp = [[0] * (n + 1) for _ in range(goal + 1)]
    dp[0][0] = 1
    for i in range(1, goal + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] * (n - j + 1) % MOD
            if j > k:
                dp[i][j] = (dp[i][j] + dp[i-1][j] * (j - k)) % MOD
    return dp[goal][n]


def count_different_palindromic_subsequences(s: str) -> int:
    """98. Count different non-empty palindromic subsequences."""
    MOD = 10**9 + 7
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                lo, hi = i + 1, j - 1
                while lo <= hi and s[lo] != s[i]:
                    lo += 1
                while lo <= hi and s[hi] != s[j]:
                    hi -= 1
                if lo > hi:
                    dp[i][j] = dp[i+1][j-1] * 2 + 2
                elif lo == hi:
                    dp[i][j] = dp[i+1][j-1] * 2 + 1
                else:
                    dp[i][j] = dp[i+1][j-1] * 2 - dp[lo+1][hi-1]
            else:
                dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
            dp[i][j] %= MOD
    return dp[0][n-1]


def find_the_shortest_superstring(words: List[str]) -> str:
    """99. Find shortest superstring containing all words."""
    n = len(words)
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m = min(len(words[i]), len(words[j]))
                for k in range(m, 0, -1):
                    if words[i].endswith(words[j][:k]):
                        overlap[i][j] = k
                        break

    dp = [[0] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask >> last & 1):
                continue
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0:
                dp[mask][last] = len(words[last])
                continue
            for prev in range(n):
                if not (prev_mask >> prev & 1):
                    continue
                val = dp[prev_mask][prev] + len(words[last]) - overlap[prev][last]
                if val > dp[mask][last] or dp[mask][last] == 0:
                    if dp[mask][last] == 0 or val < dp[mask][last]:
                        dp[mask][last] = val
                        parent[mask][last] = prev

    full = (1 << n) - 1
    last = min(range(n), key=lambda x: dp[full][x])
    path = []
    mask = full
    while last != -1:
        path.append(last)
        prev = parent[mask][last]
        mask ^= (1 << last)
        last = prev
    path.reverse()
    result = words[path[0]]
    for i in range(1, len(path)):
        result += words[path[i]][overlap[path[i-1]][path[i]]:]
    return result


def tallest_billboard(rods: List[int]) -> int:
    """100. Tallest billboard: split rods into two equal-height supports."""
    dp = {0: 0}  # diff -> max shorter side
    for r in rods:
        new_dp = dict(dp)
        for diff, short in dp.items():
            tall = short + diff
            # add rod to taller side
            d2 = diff + r
            new_dp[d2] = max(new_dp.get(d2, 0), short)
            # add rod to shorter side
            if r <= diff:
                d2 = diff - r
                new_dp[d2] = max(new_dp.get(d2, 0), short + r)
            else:
                d2 = r - diff
                new_dp[d2] = max(new_dp.get(d2, 0), tall)
        dp = new_dp
    return dp.get(0, 0)
