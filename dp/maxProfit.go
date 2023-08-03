package dp

/* 122. 买卖股票的最佳时机 II */
func maxProfit(prices []int) int {

	n := len(prices)
	// dp[i][j]表示第i天能获得的最大利润, 其中j ∈ (0,1), 0表示没有持有股票, 1表示持有股票
	// dp[i][0] = Max(dp[i-1][0], dp[i-1][1]+prices[i])
	// dp[i][1] = Max(dp[i-1][1], dp[i-1][0]-prices[i])
	// 由于dp[i]只与dp[i-1]相关, 可以使用两个变量代替数组
	dp0, dp1 := 0, -prices[0]
	for i := 1; i < n; i++ {
		dp0 = Max(dp0, dp1+prices[i])
		dp1 = Max(dp1, dp0-prices[i])
	}

	return dp0
}
