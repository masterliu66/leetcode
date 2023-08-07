package dp

import (
	"leetcode/function"
	"testing"
)

func TestLongestCommonSubsequence(t *testing.T) {

	text1, text2 := "abcde", "ace"

	ans := longestCommonSubsequence(text1, text2)

	function.AssertEqual(t, 3, ans)
}

func TestLengthOfLIS(t *testing.T) {

	nums := []int{0, 1, 0, 3, 2, 3}

	ans := lengthOfLIS(nums)

	function.AssertEqual(t, 4, ans)
}

func TestLengthOfLIS2(t *testing.T) {

	nums, k := []int{4, 2, 1, 4, 3, 4, 5, 8, 15}, 3

	ans := lengthOfLIS2(nums, k)

	function.AssertEqual(t, 5, ans)

	nums, k = []int{7, 4, 5, 1, 8, 12, 4, 7}, 5

	ans = lengthOfLIS2(nums, k)

	function.AssertEqual(t, 4, ans)
}

func TestLongestPalindrome(t *testing.T) {

	s := "tfbaabz"

	ans := longestPalindrome(s)

	function.AssertEqual(t, "baab", ans)
}

func TestLongestIdealString(t *testing.T) {

	s := "pvjcci"
	k := 4

	ans := longestIdealString(s, k)

	function.AssertEqual(t, 2, ans)
}

func TestLongestSubsequence(t *testing.T) {

	arr := []int{1, 5, 7, 8, 5, 3, 4, 2, 1}
	difference := -2

	ans := longestSubsequence(arr, difference)

	function.AssertEqual(t, 4, ans)
}

func TestWaysToStep(t *testing.T) {

	n := 1000000
	ans := waysToStep(n)
	function.AssertEqual(t, 746580045, ans)
}

func TestGenerateParenthesis(t *testing.T) {

	n := 3

	ans := generateParenthesis(n)

	function.AssertEqual(t, []string{"()()()", "()(())", "(())()", "(()())", "((()))"}, ans)
}

func TestMaxSubArray(t *testing.T) {

	nums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}

	ans := maxSubArray(nums)

	function.AssertEqual(t, 6, ans)
}

func TestCanJump(t *testing.T) {

	nums := []int{2, 3, 1, 1, 4}

	ans := canJump(nums)

	function.AssertEqual(t, true, ans)
}

func TestRob(t *testing.T) {

	nums := []int{2, 7, 9, 3, 1}

	ans := rob(nums)

	function.AssertEqual(t, 12, ans)
}

func TestMaxProfit(t *testing.T) {

	prices := []int{7, 1, 5, 3, 6, 4}

	ans := maxProfit(prices)

	function.AssertEqual(t, 7, ans)
}

func TestCoinChange(t *testing.T) {

	coins, amount := []int{1, 2, 5}, 11

	ans := coinChange(coins, amount)

	function.AssertEqual(t, 3, ans)

}

func TestJump(t *testing.T) {

	nums := []int{5, 4, 2, 2, 1, 1, 2, 3, 4, 5}

	ans := jump(nums)

	function.AssertEqual(t, 4, ans)
}

func TestTrap(t *testing.T) {

	height := []int{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}

	ans := trap(height)

	function.AssertEqual(t, 6, ans)
}

func TestUniquePaths(t *testing.T) {

	m, n := 3, 7

	ans := uniquePaths(m, n)

	function.AssertEqual(t, 28, ans)
}

func TestUniquePathsWithObstacles(t *testing.T) {

	obstacleGrid := [][]int{{0, 0}, {1, 1}, {0, 0}}

	ans := uniquePathsWithObstacles(obstacleGrid)

	function.AssertEqual(t, 0, ans)
}

func TestMinPathSum(t *testing.T) {

	grid := [][]int{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}

	ans := minPathSum(grid)

	function.AssertEqual(t, 7, ans)
}

func TestMinDistance(t *testing.T) {

	word1, word2 := "horse", "ros"

	ans := minDistance(word1, word2)

	function.AssertEqual(t, 3, ans)

	word1, word2 = "intention", "execution"

	ans = minDistance(word1, word2)

	function.AssertEqual(t, 5, ans)
}
