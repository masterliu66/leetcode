package dp

/** 2370. 最长理想子序列 */
func longestIdealString(s string, k int) int {

	// dp[i]表示第i个字母为结尾的理想字符串的最大长度
	dp := [26]int{}
	for _, c := range s {
		place := int(c - 'a')
		for _, v := range dp[Max(place-k, 0):Min(place+k+1, 26)] {
			dp[place] = Max(dp[place], v)
		}
		dp[place]++
	}

	ans := 0
	for _, v := range dp {
		ans = Max(ans, v)
	}

	return ans
}
