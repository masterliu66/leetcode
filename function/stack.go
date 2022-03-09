package function

type Stack struct {
	nums []int
}

func (stack *Stack) len() int {
	return len(stack.nums)
}

func (stack *Stack) isEmpty() bool {
	return stack.len() == 0
}

func (stack *Stack) peek() int {
	return stack.nums[stack.len()-1]
}

func (stack *Stack) pop() {
	stack.nums = stack.nums[:stack.len()-1]
}

func (stack *Stack) push(num int) {
	stack.nums = append(stack.nums, num)
}

func (stack *Stack) clear() {
	stack.nums = stack.nums[:0]
}
