package dp

/** 5. 最长回文子串 */
func longestPalindrome(s string) string {
	return longestPalindromeDp(s)
}

func longestPalindromeCenterExpansion(s string) string {

	n := len(s)
	ans := ""
	max := 0
	for i := 0; i < n; i++ {
		pre, after := i, i
		for pre > 0 && s[pre-1] == s[i] {
			pre--
		}
		for after < n-1 && s[after+1] == s[i] {
			after++
		}
		for pre > 0 && after < n-1 && s[pre-1] == s[after+1] {
			pre--
			after++
		}
		if after-pre+1 > max {
			max = after - pre + 1
			ans = s[pre : after+1]
		}
	}

	return ans
}

func longestPalindromeDp(s string) string {

	n := len(s)
	// dp[i][j]表示区间[i,j]是否为回文字符串
	dp := make([][]bool, n)
	// 初始化
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
		// 单个字符的情况
		dp[i][i] = true
		// 连续两个相同的字符的情况
		if i < n-1 && s[i] == s[i+1] {
			dp[i][i+1] = true
		}
	}
	for i := n - 1; i >= 0; i-- {
		for j := n - 1; j >= i; j-- {
			// 状态转移dp[i-1][j+1] = dp[i][j] ? s[i-1] == [j+1] : false
			if dp[i][j] && i > 0 && j < n-1 && s[i-1] == s[j+1] {
				dp[i-1][j+1] = true
			}
		}
	}

	ans := ""
	max := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if dp[i][j] && max < j-i+1 {
				max = j - i + 1
				ans = s[i : j+1]
			}
		}
	}

	return ans
}
