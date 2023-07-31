package dp

/* 55. 跳跃游戏 */
func canJump(nums []int) bool {

	n := len(nums)
	// dp[i]表示从第i格出发是否可以到达终点
	// 可以到达终点的最小下标
	minIndex := n - 1
	for i := n - 2; i >= 0; i-- {
		if nums[i] >= minIndex-i {
			minIndex = i
		}
	}
	return minIndex == 0
}
