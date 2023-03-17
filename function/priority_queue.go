package function

import "sort"

type PriorityQueue struct {
	elements []sort.Interface
}

func NewPriorityQueue(len int) PriorityQueue {
	elements := make([]sort.Interface, len)
	return PriorityQueue{elements}
}

func siftDown() {

}

func siftUp() {

}
