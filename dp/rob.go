package dp

/* 198. 打家劫舍 */
func rob(nums []int) int {

	n := len(nums)
	// dp[i]表示包含第i间房屋的最大打劫金额
	dp := make([]int, n)
	dp[0] = nums[0]
	max := nums[0]
	for i := 1; i < n; i++ {
		dp[i] = nums[i]
		for j := 0; j < i-1; j++ {
			dp[i] = Max(dp[i], nums[i]+dp[j])
		}
		max = Max(max, dp[i])
	}

	return max
}
