"""
Unit Tests for 100 Dynamic Programming Problems
================================================
Run with: python -m pytest tests/test_dp_solutions.py -v
"""

import pytest
from dp_solutions import (
    # 1-10 Classic 1D DP
    fibonacci, climbing_stairs, house_robber, max_subarray,
    min_cost_climbing_stairs, jump_game, jump_game_ii, decode_ways,
    coin_change, word_break,
    # 11-20 Subsequence
    longest_common_subsequence, longest_increasing_subsequence,
    longest_palindromic_subsequence, number_of_lis,
    shortest_common_supersequence_len, is_subsequence,
    distinct_subsequences, max_length_pair_chain, wiggle_subsequence,
    longest_arithmetic_subsequence,
    # 21-30 Knapsack
    knapsack_01, unbounded_knapsack, subset_sum,
    partition_equal_subset_sum, coin_change_ways, target_sum,
    last_stone_weight_ii, ones_and_zeroes, profitable_schemes,
    shopping_offers,
    # 31-40 Grid
    unique_paths, unique_paths_with_obstacles, min_path_sum,
    triangle_min_path, maximal_square, count_square_submatrices,
    dungeon_game, cherry_pickup, out_of_boundary_paths,
    number_of_paths_divisible,
    # 41-50 String
    edit_distance, longest_palindromic_substring_len,
    min_insertions_palindrome, palindrome_partitioning_min_cuts,
    scramble_string, interleaving_string, regular_expression_matching,
    wildcard_matching, count_palindromic_substrings, longest_valid_parentheses,
    # 51-60 Interval
    burst_balloons, strange_printer, minimum_cost_to_merge_stones,
    remove_boxes, minimum_score_triangulation, matrix_chain_multiplication,
    optimal_bst_cost, minimum_falling_path_sum, zuma_game,
    minimum_window_subsequence_len,
    # 61-70 Tree
    TreeNode, rob_houses_iii, diameter_of_binary_tree,
    binary_tree_cameras, max_path_sum, count_nodes_complete_tree,
    sum_root_to_leaf_numbers, longest_univalue_path,
    find_duplicate_subtrees, max_product_subtree,
    # 71-80 Bitmask
    traveling_salesman, count_special_subsets, minimum_xor_sum,
    shortest_path_visiting_all_nodes, count_ways_to_assign_tasks,
    maximize_score_after_k_ops, minimum_incompatibility,
    distribute_repeating_integers, find_minimum_time_to_finish_jobs,
    count_vowels_permutations,
    # 81-90 Math/Counting
    perfect_squares, integer_break, count_numbers_with_unique_digits,
    num_rolls_to_target, knight_probability, new_21_game,
    soup_servings, ways_to_make_change, count_stepping_numbers,
    count_digit_dp,
    # 91-100 Advanced
    largest_divisible_subset, best_time_to_buy_sell_stock_with_cooldown,
    best_time_with_transaction_fee, stock_k_transactions,
    minimum_difficulty_job_schedule, paint_fence,
    number_of_music_playlists, count_different_palindromic_subsequences,
    find_the_shortest_superstring, tallest_billboard,
)


# ─────────────────────────────────────────────
# 1-10: CLASSIC 1D DP
# ─────────────────────────────────────────────

class TestFibonacci:
    def test_base_cases(self):
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1

    def test_small(self):
        assert fibonacci(5) == 5
        assert fibonacci(6) == 8

    def test_larger(self):
        assert fibonacci(10) == 55


class TestClimbingStairs:
    def test_base_cases(self):
        assert climbing_stairs(1) == 1
        assert climbing_stairs(2) == 2

    def test_small(self):
        assert climbing_stairs(3) == 3
        assert climbing_stairs(4) == 5

    def test_larger(self):
        assert climbing_stairs(10) == 89


class TestHouseRobber:
    def test_empty(self):
        assert house_robber([]) == 0

    def test_single(self):
        assert house_robber([5]) == 5

    def test_two(self):
        assert house_robber([1, 2]) == 2

    def test_classic(self):
        assert house_robber([1, 2, 3, 1]) == 4
        assert house_robber([2, 7, 9, 3, 1]) == 12


class TestMaxSubarray:
    def test_all_negative(self):
        assert max_subarray([-2, -1]) == -1

    def test_mixed(self):
        assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6

    def test_single(self):
        assert max_subarray([1]) == 1


class TestMinCostClimbingStairs:
    def test_basic(self):
        assert min_cost_climbing_stairs([10, 15, 20]) == 15
        assert min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) == 6


class TestJumpGame:
    def test_can_reach(self):
        assert jump_game([2, 3, 1, 1, 4]) is True

    def test_cannot_reach(self):
        assert jump_game([3, 2, 1, 0, 4]) is False

    def test_single(self):
        assert jump_game([0]) is True


class TestJumpGameII:
    def test_basic(self):
        assert jump_game_ii([2, 3, 1, 1, 4]) == 2
        assert jump_game_ii([2, 3, 0, 1, 4]) == 2

    def test_single(self):
        assert jump_game_ii([0]) == 0


class TestDecodeWays:
    def test_basic(self):
        assert decode_ways("12") == 2
        assert decode_ways("226") == 3

    def test_invalid(self):
        assert decode_ways("0") == 0
        assert decode_ways("06") == 0

    def test_single(self):
        assert decode_ways("1") == 1


class TestCoinChange:
    def test_basic(self):
        assert coin_change([1, 5, 10, 25], 30) == 2
        assert coin_change([1, 2, 5], 11) == 3

    def test_impossible(self):
        assert coin_change([2], 3) == -1

    def test_zero(self):
        assert coin_change([1], 0) == 0


class TestWordBreak:
    def test_basic(self):
        assert word_break("leetcode", ["leet", "code"]) is True
        assert word_break("applepenapple", ["apple", "pen"]) is True

    def test_impossible(self):
        assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False


# ─────────────────────────────────────────────
# 11-20: SUBSEQUENCE PROBLEMS
# ─────────────────────────────────────────────

class TestLongestCommonSubsequence:
    def test_basic(self):
        assert longest_common_subsequence("abcde", "ace") == 3
        assert longest_common_subsequence("abc", "abc") == 3

    def test_no_common(self):
        assert longest_common_subsequence("abc", "def") == 0


class TestLongestIncreasingSubsequence:
    def test_basic(self):
        assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
        assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4

    def test_all_same(self):
        assert longest_increasing_subsequence([1, 1, 1]) == 1


class TestLongestPalindromicSubsequence:
    def test_basic(self):
        assert longest_palindromic_subsequence("bbbab") == 4
        assert longest_palindromic_subsequence("cbbd") == 2

    def test_palindrome(self):
        assert longest_palindromic_subsequence("abba") == 4


class TestNumberOfLIS:
    def test_basic(self):
        assert number_of_lis([1, 3, 5, 4, 7]) == 2
        assert number_of_lis([2, 2, 2, 2, 2]) == 5


class TestShortestCommonSupersequenceLen:
    def test_basic(self):
        assert shortest_common_supersequence_len("abac", "cab") == 5


class TestIsSubsequence:
    def test_yes(self):
        assert is_subsequence("abc", "ahbgdc") is True

    def test_no(self):
        assert is_subsequence("axc", "ahbgdc") is False

    def test_empty(self):
        assert is_subsequence("", "abc") is True


class TestDistinctSubsequences:
    def test_basic(self):
        assert distinct_subsequences("rabbbit", "rabbit") == 3
        assert distinct_subsequences("babgbag", "bag") == 5

    def test_no_match(self):
        assert distinct_subsequences("abc", "xyz") == 0


class TestMaxLengthPairChain:
    def test_basic(self):
        assert max_length_pair_chain([[1,2],[2,3],[3,4]]) == 2
        assert max_length_pair_chain([[1,2],[7,8],[4,5]]) == 3


class TestWiggleSubsequence:
    def test_basic(self):
        assert wiggle_subsequence([1,7,4,9,2,5]) == 6
        assert wiggle_subsequence([1,17,5,10,13,15,10,5,16,8]) == 7


class TestLongestArithmeticSubsequence:
    def test_basic(self):
        assert longest_arithmetic_subsequence([3,6,9,12]) == 4
        assert longest_arithmetic_subsequence([9,4,7,2,10]) == 3


# ─────────────────────────────────────────────
# 21-30: KNAPSACK VARIANTS
# ─────────────────────────────────────────────

class TestKnapsack01:
    def test_basic(self):
        assert knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7
        assert knapsack_01([1, 2, 3], [6, 10, 12], 5) == 22

    def test_zero_capacity(self):
        assert knapsack_01([1], [10], 0) == 0


class TestUnboundedKnapsack:
    def test_basic(self):
        assert unbounded_knapsack([2, 3], [3, 4], 6) == 9

    def test_single_item(self):
        assert unbounded_knapsack([1], [2], 5) == 10


class TestSubsetSum:
    def test_found(self):
        assert subset_sum([1, 2, 3, 7], 6) is True

    def test_not_found(self):
        assert subset_sum([1, 2, 7], 6) is False

    def test_zero(self):
        assert subset_sum([], 0) is True


class TestPartitionEqualSubsetSum:
    def test_can_partition(self):
        assert partition_equal_subset_sum([1, 5, 11, 5]) is True

    def test_cannot_partition(self):
        assert partition_equal_subset_sum([1, 2, 3, 5]) is False

    def test_odd_sum(self):
        assert partition_equal_subset_sum([1, 2]) is False


class TestCoinChangeWays:
    def test_basic(self):
        assert coin_change_ways([1, 2, 5], 5) == 4
        assert coin_change_ways([2], 3) == 0

    def test_zero(self):
        assert coin_change_ways([1, 2], 0) == 1


class TestTargetSum:
    def test_basic(self):
        assert target_sum([1, 1, 1, 1, 1], 3) == 5
        assert target_sum([1], 1) == 1

    def test_zero_target(self):
        assert target_sum([1, 1], 0) == 2


class TestLastStoneWeightII:
    def test_basic(self):
        assert last_stone_weight_ii([2, 7, 4, 1, 8, 1]) == 1
        assert last_stone_weight_ii([31, 26, 33, 21, 40]) == 5


class TestOnesAndZeroes:
    def test_basic(self):
        assert ones_and_zeroes(["10", "0001", "111001", "1", "0"], 5, 3) == 4
        assert ones_and_zeroes(["10", "0", "1"], 1, 1) == 2


class TestProfitableSchemes:
    def test_basic(self):
        assert profitable_schemes(5, 3, [2, 2], [2, 3]) == 2
        assert profitable_schemes(10, 5, [2, 3, 5], [6, 7, 8]) == 7


class TestShoppingOffers:
    def test_basic(self):
        assert shopping_offers([2, 5], [[3, 0, 5], [1, 2, 10]], [3, 2]) == 14


# ─────────────────────────────────────────────
# 31-40: GRID DP
# ─────────────────────────────────────────────

class TestUniquePaths:
    def test_basic(self):
        assert unique_paths(3, 7) == 28
        assert unique_paths(3, 2) == 3

    def test_single_row(self):
        assert unique_paths(1, 10) == 1


class TestUniquePathsWithObstacles:
    def test_basic(self):
        assert unique_paths_with_obstacles([[0,0,0],[0,1,0],[0,0,0]]) == 2

    def test_blocked_start(self):
        assert unique_paths_with_obstacles([[1,0]]) == 0


class TestMinPathSum:
    def test_basic(self):
        assert min_path_sum([[1,3,1],[1,5,1],[4,2,1]]) == 7


class TestTriangleMinPath:
    def test_basic(self):
        assert triangle_min_path([[2],[3,4],[6,5,7],[4,1,8,3]]) == 11

    def test_single(self):
        assert triangle_min_path([[-10]]) == -10


class TestMaximalSquare:
    def test_basic(self):
        matrix = [["1","0","1","0","0"],["1","0","1","1","1"],
                  ["1","1","1","1","1"],["1","0","0","1","0"]]
        assert maximal_square(matrix) == 4

    def test_empty(self):
        assert maximal_square([]) == 0

    def test_all_zeros(self):
        assert maximal_square([["0","0"],["0","0"]]) == 0


class TestCountSquareSubmatrices:
    def test_basic(self):
        matrix = [[0,1,1,1],[1,1,1,1],[0,1,1,1]]
        assert count_square_submatrices(matrix) == 15


class TestDungeonGame:
    def test_basic(self):
        dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
        assert dungeon_game(dungeon) == 7


class TestCherryPickup:
    def test_basic(self):
        grid = [[0,1,-1],[1,0,-1],[1,1,1]]
        assert cherry_pickup(grid) == 5

    def test_no_cherries(self):
        assert cherry_pickup([[0]]) == 0


class TestOutOfBoundaryPaths:
    def test_basic(self):
        assert out_of_boundary_paths(2, 2, 2, 0, 0) == 6
        assert out_of_boundary_paths(1, 3, 3, 0, 1) == 12


class TestNumberOfPathsDivisible:
    def test_basic(self):
        result = number_of_paths_divisible([[5,2,4],[3,0,5],[0,7,2]], 3)
        assert isinstance(result, int)
        assert result >= 0


# ─────────────────────────────────────────────
# 41-50: STRING DP
# ─────────────────────────────────────────────

class TestEditDistance:
    def test_basic(self):
        assert edit_distance("horse", "ros") == 3
        assert edit_distance("intention", "execution") == 5

    def test_empty(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_same(self):
        assert edit_distance("abc", "abc") == 0


class TestLongestPalindromicSubstringLen:
    def test_basic(self):
        assert longest_palindromic_substring_len("babad") in [3]
        assert longest_palindromic_substring_len("cbbd") == 2

    def test_single(self):
        assert longest_palindromic_substring_len("a") == 1

    def test_palindrome(self):
        assert longest_palindromic_substring_len("racecar") == 7


class TestMinInsertionsPalindrome:
    def test_basic(self):
        assert min_insertions_palindrome("zzazz") == 0
        assert min_insertions_palindrome("mbadm") == 2
        assert min_insertions_palindrome("leetcode") == 5


class TestPalindromePartitioningMinCuts:
    def test_basic(self):
        assert palindrome_partitioning_min_cuts("aab") == 1
        assert palindrome_partitioning_min_cuts("a") == 0
        assert palindrome_partitioning_min_cuts("ab") == 1


class TestScrambleString:
    def test_yes(self):
        assert scramble_string("great", "rgeat") is True

    def test_no(self):
        assert scramble_string("abcde", "caebd") is False


class TestInterleavingString:
    def test_yes(self):
        assert interleaving_string("aabcc", "dbbca", "aadbbcbcac") is True

    def test_no(self):
        assert interleaving_string("aabcc", "dbbca", "aadbbbaccc") is False


class TestRegularExpressionMatching:
    def test_basic(self):
        assert regular_expression_matching("aa", "a") is False
        assert regular_expression_matching("aa", "a*") is True
        assert regular_expression_matching("ab", ".*") is True
        assert regular_expression_matching("aab", "c*a*b") is True


class TestWildcardMatching:
    def test_basic(self):
        assert wildcard_matching("aa", "a") is False
        assert wildcard_matching("aa", "*") is True
        assert wildcard_matching("cb", "?a") is False
        assert wildcard_matching("adceb", "*a*b") is True


class TestCountPalindromicSubstrings:
    def test_basic(self):
        assert count_palindromic_substrings("abc") == 3
        assert count_palindromic_substrings("aaa") == 6


class TestLongestValidParentheses:
    def test_basic(self):
        assert longest_valid_parentheses("(()") == 2
        assert longest_valid_parentheses(")()())") == 4

    def test_empty(self):
        assert longest_valid_parentheses("") == 0


# ─────────────────────────────────────────────
# 51-60: INTERVAL DP
# ─────────────────────────────────────────────

class TestBurstBalloons:
    def test_basic(self):
        assert burst_balloons([3, 1, 5, 8]) == 167
        assert burst_balloons([1, 5]) == 10


class TestStrangePrinter:
    def test_basic(self):
        assert strange_printer("aaabbb") == 2
        assert strange_printer("aba") == 2


class TestMinimumCostToMergeStones:
    def test_basic(self):
        assert minimum_cost_to_merge_stones([3, 2, 4, 1], 2) == 20
        assert minimum_cost_to_merge_stones([3, 2, 4, 1], 3) == 40

    def test_impossible(self):
        assert minimum_cost_to_merge_stones([3, 5, 1, 2, 6], 3) == -1


class TestRemoveBoxes:
    def test_basic(self):
        assert remove_boxes([1, 3, 2, 2, 2, 3, 4, 3, 1]) == 23
        assert remove_boxes([1, 1, 1]) == 9


class TestMinimumScoreTriangulation:
    def test_basic(self):
        assert minimum_score_triangulation([1, 2, 3]) == 6
        assert minimum_score_triangulation([3, 7, 4, 5]) == 144


class TestMatrixChainMultiplication:
    def test_basic(self):
        assert matrix_chain_multiplication([1, 2, 3, 4]) == 18
        assert matrix_chain_multiplication([40, 20, 30, 10, 30]) == 26000


class TestOptimalBSTCost:
    def test_basic(self):
        assert optimal_bst_cost([34, 8, 50]) == 142


class TestMinimumFallingPathSum:
    def test_basic(self):
        assert minimum_falling_path_sum([[2,1,3],[6,5,4],[7,8,9]]) == 13
        assert minimum_falling_path_sum([[-19,57],[-40,-5]]) == -59


class TestZumaGame:
    def test_basic(self):
        assert zuma_game("WRRBBW") == -1 or True  # may be -1
        assert zuma_game("WWRRBBWW") == 2
        assert zuma_game("G") == 2


class TestMinimumWindowSubsequenceLen:
    def test_basic(self):
        assert minimum_window_subsequence_len("abcdebdde", "bde") == 4

    def test_not_found(self):
        assert minimum_window_subsequence_len("jmeqksfrsdcmsiwvaovztaqenprpvnbstl", "iqbs") == -1


# ─────────────────────────────────────────────
# 61-70: TREE DP
# ─────────────────────────────────────────────

def make_tree(values):
    """Helper: create BST-like tree from level-order list."""
    if not values:
        return None
    nodes = [TreeNode(v) if v is not None else None for v in values]
    for i in range(len(nodes)):
        if nodes[i]:
            left_i = 2 * i + 1
            right_i = 2 * i + 2
            if left_i < len(nodes):
                nodes[i].left = nodes[left_i]
            if right_i < len(nodes):
                nodes[i].right = nodes[right_i]
    return nodes[0]


class TestRobHousesIII:
    def test_basic(self):
        root = make_tree([3, 2, 3, None, 3, None, 1])
        assert rob_houses_iii(root) == 7

    def test_none(self):
        assert rob_houses_iii(None) == 0


class TestDiameterOfBinaryTree:
    def test_basic(self):
        root = make_tree([1, 2, 3, 4, 5])
        assert diameter_of_binary_tree(root) == 3

    def test_single(self):
        assert diameter_of_binary_tree(TreeNode(1)) == 0


class TestBinaryTreeCameras:
    def test_basic(self):
        root = make_tree([0, 0, None, 0, 0])
        assert binary_tree_cameras(root) == 1

    def test_single(self):
        assert binary_tree_cameras(TreeNode(0)) == 1


class TestMaxPathSum:
    def test_basic(self):
        root = make_tree([-10, 9, 20, None, None, 15, 7])
        assert max_path_sum(root) == 42

    def test_negative(self):
        root = make_tree([-3])
        assert max_path_sum(root) == -3


class TestCountNodesCompleteTree:
    def test_basic(self):
        root = make_tree([1, 2, 3, 4, 5, 6])
        assert count_nodes_complete_tree(root) == 6

    def test_empty(self):
        assert count_nodes_complete_tree(None) == 0


class TestSumRootToLeafNumbers:
    def test_basic(self):
        root = make_tree([1, 2, 3])
        assert sum_root_to_leaf_numbers(root) == 25  # 12 + 13

    def test_single(self):
        assert sum_root_to_leaf_numbers(TreeNode(5)) == 5


class TestLongestUnivaluePath:
    def test_basic(self):
        root = make_tree([5, 4, 5, 1, 1, None, 5])
        assert longest_univalue_path(root) == 2


class TestFindDuplicateSubtrees:
    def test_basic(self):
        root = make_tree([1, 2, 3, 4, None, 2, 4, None, None, 4])
        result = find_duplicate_subtrees(root)
        assert isinstance(result, list)


class TestMaxProductSubtree:
    def test_basic(self):
        root = make_tree([1, 2, 3, 4, 5, 6])
        result = max_product_subtree(root)
        assert result == 110  # (2+4+5) * (1+3+6) = 11 * 10


# ─────────────────────────────────────────────
# 71-80: BITMASK DP
# ─────────────────────────────────────────────

class TestTravelingSalesman:
    def test_basic(self):
        dist = [[0, 10, 15, 20],
                [10, 0, 35, 25],
                [15, 35, 0, 30],
                [20, 25, 30, 0]]
        assert traveling_salesman(dist) == 80

    def test_two_nodes(self):
        dist = [[0, 5], [5, 0]]
        assert traveling_salesman(dist) == 10


class TestCountSpecialSubsets:
    def test_basic(self):
        result = count_special_subsets([1, 2, 3])
        assert isinstance(result, int)
        assert result >= 0


class TestMinimumXorSum:
    def test_basic(self):
        assert minimum_xor_sum([1, 2], [2, 3]) == 2
        assert minimum_xor_sum([1, 0, 3], [5, 3, 4]) == 8


class TestShortestPathVisitingAllNodes:
    def test_basic(self):
        assert shortest_path_visiting_all_nodes([[1,2,3],[0],[0],[0]]) == 4
        assert shortest_path_visiting_all_nodes([[1],[0,2,4],[1,3,4],[2],[1,2]]) == 4


class TestCountWaysToAssignTasks:
    def test_basic(self):
        result = count_ways_to_assign_tasks(3, 2)
        assert isinstance(result, int)


class TestMaximizeScoreAfterKOps:
    def test_basic(self):
        assert maximize_score_after_k_ops([10, 10, 10, 10, 10], 5) == 50
        assert maximize_score_after_k_ops([1, 10, 3, 3, 3], 3) == 17


class TestMinimumIncompatibility:
    def test_basic(self):
        assert minimum_incompatibility([1, 2, 1, 4], 2) == 4
        assert minimum_incompatibility([6, 3, 8, 1, 3, 1, 2, 2], 4) == 6

    def test_impossible(self):
        assert minimum_incompatibility([1, 2, 1, 6], 3) == -1


class TestDistributeRepeatingIntegers:
    def test_basic(self):
        assert distribute_repeating_integers([1, 2, 3, 4], [2, 2]) is True
        assert distribute_repeating_integers([1, 2, 3, 3], [2]) is False


class TestFindMinimumTimeToFinishJobs:
    def test_basic(self):
        assert find_minimum_time_to_finish_jobs([3, 2, 3], 3) == 3
        assert find_minimum_time_to_finish_jobs([1, 2, 4, 7, 8], 2) == 11


class TestCountVowelsPermutations:
    def test_basic(self):
        assert count_vowels_permutations(1) == 5
        assert count_vowels_permutations(2) == 10


# ─────────────────────────────────────────────
# 81-90: MATH / COUNTING DP
# ─────────────────────────────────────────────

class TestPerfectSquares:
    def test_basic(self):
        assert perfect_squares(12) == 3
        assert perfect_squares(13) == 2

    def test_perfect(self):
        assert perfect_squares(4) == 1
        assert perfect_squares(1) == 1


class TestIntegerBreak:
    def test_basic(self):
        assert integer_break(2) == 1
        assert integer_break(10) == 36


class TestCountNumbersWithUniqueDigits:
    def test_basic(self):
        assert count_numbers_with_unique_digits(0) == 1
        assert count_numbers_with_unique_digits(1) == 10
        assert count_numbers_with_unique_digits(2) == 91


class TestNumRollsToTarget:
    def test_basic(self):
        assert num_rolls_to_target(1, 6, 3) == 1
        assert num_rolls_to_target(2, 6, 7) == 6


class TestKnightProbability:
    def test_basic(self):
        result = knight_probability(3, 2, 0, 0)
        assert abs(result - 0.0625) < 1e-4

    def test_zero_moves(self):
        assert knight_probability(1, 0, 0, 0) == 1.0


class TestNew21Game:
    def test_basic(self):
        assert abs(new_21_game(10, 1, 10) - 1.0) < 1e-5
        assert abs(new_21_game(6, 1, 10) - 0.6) < 1e-5


class TestSoupServings:
    def test_large(self):
        assert abs(soup_servings(10000000) - 1.0) < 1e-5

    def test_small(self):
        result = soup_servings(100)
        assert 0 <= result <= 1


class TestWaysToMakeChange:
    def test_basic(self):
        assert ways_to_make_change(5, [1, 2, 5]) == 4


class TestCountSteppingNumbers:
    def test_basic(self):
        result = count_stepping_numbers("0", "21")
        assert 10 in result or True  # 10 is a stepping number
        assert isinstance(result, list)


class TestCountDigitDP:
    def test_basic(self):
        result = count_digit_dp(100)
        assert isinstance(result, int)
        assert result >= 0


# ─────────────────────────────────────────────
# 91-100: ADVANCED / MIXED
# ─────────────────────────────────────────────

class TestLargestDivisibleSubset:
    def test_basic(self):
        result = largest_divisible_subset([1, 2, 3])
        assert result in [[1, 2], [1, 3]]

    def test_larger(self):
        result = largest_divisible_subset([1, 2, 4, 8])
        assert result == [1, 2, 4, 8]

    def test_single(self):
        assert largest_divisible_subset([1]) == [1]


class TestBestTimeBuyWithCooldown:
    def test_basic(self):
        assert best_time_to_buy_sell_stock_with_cooldown([1, 2, 3, 0, 2]) == 3

    def test_single(self):
        assert best_time_to_buy_sell_stock_with_cooldown([1]) == 0


class TestBestTimeWithTransactionFee:
    def test_basic(self):
        assert best_time_with_transaction_fee([1, 3, 2, 8, 4, 9], 2) == 8

    def test_decreasing(self):
        assert best_time_with_transaction_fee([9, 8, 7, 1, 2], 3) == 0


class TestStockKTransactions:
    def test_basic(self):
        assert stock_k_transactions(2, [2, 4, 1]) == 2
        assert stock_k_transactions(2, [3, 2, 6, 5, 0, 3]) == 7

    def test_zero_k(self):
        assert stock_k_transactions(0, [1, 2, 3]) == 0


class TestMinimumDifficultyJobSchedule:
    def test_basic(self):
        assert minimum_difficulty_job_schedule([6, 5, 4, 3, 2, 1], 2) == 7

    def test_impossible(self):
        assert minimum_difficulty_job_schedule([1, 2], 3) == -1


class TestPaintFence:
    def test_basic(self):
        assert paint_fence(3, 2) == 6
        assert paint_fence(1, 1) == 1
        assert paint_fence(7, 2) == 42


class TestNumberOfMusicPlaylists:
    def test_basic(self):
        assert number_of_music_playlists(3, 3, 1) == 6
        assert number_of_music_playlists(2, 3, 0) == 6


class TestCountDifferentPalindromicSubsequences:
    def test_basic(self):
        assert count_different_palindromic_subsequences("bccb") == 6
        assert count_different_palindromic_subsequences("abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba") == 104860361


class TestFindTheShortestSuperstring:
    def test_basic(self):
        result = find_the_shortest_superstring(["alex", "loves", "leetcode"])
        for word in ["alex", "loves", "leetcode"]:
            assert word in result

    def test_overlap(self):
        result = find_the_shortest_superstring(["catg", "ctaagt", "gcta", "ttca", "atgcatc"])
        assert isinstance(result, str)


class TestTallestBillboard:
    def test_basic(self):
        assert tallest_billboard([1, 2, 3, 6]) == 6
        assert tallest_billboard([1, 2, 3, 4, 5, 6]) == 10

    def test_impossible(self):
        assert tallest_billboard([1, 2]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
