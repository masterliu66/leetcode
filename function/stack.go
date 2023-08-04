package function

type Stack struct {
	nums []int
}

func (stack *Stack) Len() int {
	return len(stack.nums)
}

func (stack *Stack) IsEmpty() bool {
	return stack.Len() == 0
}

func (stack *Stack) Peek() int {
	return stack.PeekLast()
}

func (stack *Stack) PeekFirst() int {
	return stack.nums[0]
}

func (stack *Stack) PeekLast() int {
	return stack.nums[stack.Len()-1]
}

func (stack *Stack) Pop() int {
	num := stack.Peek()
	stack.nums = stack.nums[:stack.Len()-1]
	return num
}

func (stack *Stack) Push(num int) {
	stack.nums = append(stack.nums, num)
}

func (stack *Stack) Clear() {
	stack.nums = stack.nums[:0]
}
