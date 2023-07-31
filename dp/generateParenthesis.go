package dp

/* 22. 括号生成 */
func generateParenthesis(n int) []string {

	// dp[i]表示对数为i时所有有效的括号组合
	dp := map[int][]string{0: {""}, 1: {"()"}}
	for i := 2; i <= n; i++ {
		dp[i] = []string{}
		// 第i对括号组合可以写成(a)b形式, a和b表示剩余的i-1对括号组合, 其中b=i-1-a
		for j := 0; j < i; j++ {
			for _, inner := range dp[j] {
				for _, outer := range dp[i-1-j] {
					dp[i] = append(dp[i], "("+inner+")"+outer)
				}
			}
		}
	}

	return dp[n]
}
