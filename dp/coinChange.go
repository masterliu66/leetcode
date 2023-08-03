package dp

import (
	"math"
)

/* 322. 零钱兑换 */
func coinChange(coins []int, amount int) int {

	if amount == 0 {
		return 0
	}

	// dp[i]表示总价格为i时的最小硬币数量
	// dp[i] = dp[i - j] + 1, 其中j表示硬币币值
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt32
		for _, val := range coins {
			if val > i || dp[i-val] == -1 {
				continue
			}
			dp[i] = Min(dp[i], dp[i-val]+1)
		}
		if dp[i] == math.MaxInt32 {
			dp[i] = -1
		}
	}

	return dp[amount]
}
