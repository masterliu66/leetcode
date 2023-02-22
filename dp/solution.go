package dp

import (
	"leetcode/function"
)

/** 1143. 最长公共子序列 */
func longestCommonSubsequence(text1 string, text2 string) int {

	// dp[i][j]表示text1前i个字符、text2前j个字符时公共子序列的最大长度
	dp := make([][]int, len(text1) + 1)
	dp[0] = make([]int, len(text2) + 1)
	for i := 1; i <= len(text1); i++ {
		dp[i] = make([]int, len(text2) + 1)
		for j := 1; j <= len(text2); j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = function.Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	return dp[len(text1)][len(text2)]
}

/** 300. 最长递增子序列 */
func lengthOfLIS(nums []int) int {

	n := len(nums)
	// dp[i]表示长度为i的递增子序列的最小值，dp单调递增
	dp := make([]int, n + 1)
	// 初始化dp
	dp[1] = nums[0]
	// maxLen表示遍历过程中当前递增子序列的最大长度
	maxLen := 1
	for _, num := range nums[1:] {
		if dp[maxLen] < num {
			maxLen++
			dp[maxLen] = num
			continue
		}
		// 二分查找小于num的递增子序列的最大长度
		left, right := 1, maxLen
		for left < right {
			mid := (left + right + 1) >> 1
			if dp[mid] < num {
				left = mid
			} else {
				right = mid - 1
			}
		}
		if dp[left] < num {
			dp[left + 1] = num
		} else {
			// 所有数都比num大, 长度为1的递增子序列的最小值为num
			dp[1] = num
		}
	}

	return maxLen
}


/** 2407. 最长递增子序列 II */
func lengthOfLIS2(nums []int, k int) int {
	return 0
}
