package dp

import "leetcode/function"

/* 42. 接雨水 */
func trap(height []int) int {

	n := len(height)
	// dpl[i]表示下标为i的元素左边的最大高度, dpr[i]表示下标为i的元素右边的最大高度
	dpl, dpr := make([]int, n), make([]int, n)
	dpl[0] = height[0]
	for i := 1; i < n; i++ {
		dpl[i] = Max(dpl[i-1], height[i])
	}

	dpr[n-1] = height[n-1]
	for i := n - 2; i >= 0; i-- {
		dpr[i] = Max(dpr[i+1], height[i])
	}

	ans := 0
	for i := 0; i < n; i++ {
		ans += Min(dpl[i], dpr[i]) - height[i]
	}

	return ans
}

func solutionUsingStack(height []int) int {

	stack := &function.Stack{}

	ans := 0
	for i, val := range height {
		// 栈空, 或高度单调递减时直接入栈
		for !stack.IsEmpty() && val > height[stack.Peek()] {
			// 右边界为i, 途经栈顶元素, 左边界为栈顶的下一个元素
			top := stack.Pop()
			if stack.IsEmpty() {
				break
			}
			left := stack.Peek()
			w := i - left - 1
			h := Min(height[left], val) - height[top]
			ans += w * h
		}
		stack.Push(i)
	}

	return ans
}
