package dp

/* 53. 最大子数组和 */
func maxSubArray(nums []int) int {

	n := len(nums)
	// dp[i]表示以i为结尾的连续子数组的最大元素和
	dp := make([]int, n)
	dp[0] = nums[0]
	ans := nums[0]
	for i := 1; i < n; i++ {
		dp[i] = Max(nums[i], dp[i-1]+nums[i])
		ans = Max(ans, dp[i])
	}
	return ans
}
