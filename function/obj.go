package function

import "math"

var NULL = math.MinInt

type Node struct {
	Val int
	Children []*Node
}

func newNode(num int) *Node {

	if num == NULL {
		panic("null is not allowed")
	}

	return &Node{num, nil}
}

func NewTree(nums []int) *Node {

	root := newNode(nums[0])
	queue := Queue{}
	queue.offer(root)
	var node *Node
	for _, num := range nums[1:] {
		if num == -1 {
			node = queue.poll().(*Node)
		} else {
			if node.Children == nil {
				node.Children = []*Node{}
			}
			child := newNode(num)
			node.Children = append(node.Children, child)
			queue.offer(child)
		}
	}

	return root
}