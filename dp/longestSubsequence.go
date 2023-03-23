package dp

/** 1218. 最长定差子序列 */
func longestSubsequence(arr []int, difference int) int {

	// dp[i]表示以arr[i]为结尾的最长的等差子序列的长度
	dp := map[int]int{}
	ans := 1
	for _, num := range arr {
		dp[num] = dp[num-difference] + 1
		if dp[num] > ans {
			ans = dp[num]
		}
	}

	return ans
}
