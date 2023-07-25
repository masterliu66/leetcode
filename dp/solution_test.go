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
