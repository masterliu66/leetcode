package array

/** 453. 最小操作次数使数组元素相等 */
func minMoves(nums []int) int {

	min := nums[0]
	for _, num := range nums {
		min = Min(min, num)
	}

	ans := 0
	for _, num := range nums {
		ans += num - min
	}

	return ans
}
