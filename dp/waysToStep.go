package dp

/* 面试题 08.01. 三步问题 */
func waysToStep(n int) int {
	mod := 1000000007
	// 状态转移: dp[i] = dp[i-3] + dp[i-2] + dp[i-1]
	dp := []int {1, 2, 4, 7}
	if n < 4 {
		return dp[n-1]
	}
	for i := 0; i < n - 3; i++ {
		dp[i % 3] = (dp[0] + (dp[1] + dp[2]) % mod) % mod
	}
	return dp[(n - 4) % 3]
}
