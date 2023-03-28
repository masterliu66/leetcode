package array

import (
	"leetcode/function"
	"testing"
)

func TestMinMoves(t *testing.T) {

	nums := []int{1, 2, 3}

	ans := minMoves(nums)

	function.AssertEqual(t, 3, ans)
}
