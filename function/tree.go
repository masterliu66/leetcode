package function

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func newTreeNode(num int) *TreeNode {

	if num == NULL {
		panic("null is not allowed")
	}

	return &TreeNode{num, nil, nil}
}

func NewBinaryTree(nums []int) *TreeNode {

	root := newTreeNode(nums[0])
	queue := Queue{}
	queue.offer(root)
	var node *TreeNode
	for i, n := 1, len(nums); i < n; i++ {
		node = queue.poll().(*TreeNode)
		left := nums[i]
		if left != NULL {
			node.Left = newTreeNode(left)
			queue.offer(node.Left)
		}
		if i < n - 1 {
			i++
			right := nums[i]
			if right != NULL {
				node.Right = newTreeNode(right)
				queue.offer(node.Right)
			}
		}
	}

	return root
}
