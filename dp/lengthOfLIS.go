package dp

/** 2407. 最长递增子序列 II */
func lengthOfLIS2(nums []int, k int) int {

	max := nums[0]
	for _, num := range nums[1:] {
		max = Max(max, num)
	}

	// dp[i][j]表示前i个元素，最大值为j时递增子序列的最大长度
	// 由于dp[i]只会从dp[i-1]转换过来, 所以可以省略dp的第一个维度
	lisDp = make([]int, max * 4)
	for _, num := range nums {
		if num == 1 {
			// 最小值最大长度为1
			update(1, 1, max, 1, 1, 1)
			continue
		}
		// 满足条件下的最大长度 = 线段树区间[num - k, num - 1]的最大值
		length := query(1, 1, max, Max(num - k, 1), num - 1) + 1
		// 更新线段树区间[num, num]
		update(1, 1, max, num, num, length)
	}

	return lisDp[1]
}

var lisDp []int

func update(root int, start int, end int, left int, right int, val int)  {
	if start == end {
		lisDp[root] = val
		return
	}
	mid := start + ((end - start) >> 1)
	if left <= mid {
		update(root * 2, start, mid, left, right, val)
	}
	if right > mid {
		update(root * 2 + 1, mid + 1, end, left, right, val)
	}
	lisDp[root] = Max(lisDp[root * 2], lisDp[root * 2 + 1])
}

func query(root int, start int, end int, left int, right int) int {
	if start >= left && end <= right {
		return lisDp[root]
	}
	mid := start + ((end - start) >> 1)
	ans := 0
	if left <= mid {
		ans = query(root * 2, start, mid, left, right)
	}
	if right > mid {
		ans = Max(ans, query(root * 2 + 1, mid + 1, end, left, right))
	}
	return ans
}

func Max(a, b int) int {
	if a < b {
		return b
	}
	return a
}