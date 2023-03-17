package dp

/** 1143. 最长公共子序列 */
func longestCommonSubsequence(text1 string, text2 string) int {

	// dp[i][j]表示text1前i个字符、text2前j个字符时公共子序列的最大长度
	dp := make([][]int, len(text1)+1)
	dp[0] = make([]int, len(text2)+1)
	for i := 1; i <= len(text1); i++ {
		dp[i] = make([]int, len(text2)+1)
		for j := 1; j <= len(text2); j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	return dp[len(text1)][len(text2)]
}
