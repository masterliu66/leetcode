package dp

/** 300. 最长递增子序列 */
func lengthOfLIS(nums []int) int {

	n := len(nums)
	// dp[i]表示长度为i的递增子序列的最小值，dp单调递增
	dp := make([]int, n+1)
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
			dp[left+1] = num
		} else {
			// 所有数都比num大, 长度为1的递增子序列的最小值为num
			dp[1] = num
		}
	}

	return maxLen
}
