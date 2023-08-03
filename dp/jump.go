package dp

import (
	"math"
)

/* 45. 跳跃游戏 II */
func jump(nums []int) int {
	n := len(nums)
	// dp[i]表示到达第i个索引的最小跳跃次数
	// dp[i] = dp[j] + 1, 其中nums[j] >= i - j
	dp := make([]int, n)
	dp[0] = 0
	for i := 1; i < n; i++ {
		dp[i] = math.MaxInt32
		for j := 0; j < i; j++ {
			if nums[j] >= i-j {
				dp[i] = Min(dp[i], dp[j]+1)
			}
		}
	}

	return dp[n-1]
}
