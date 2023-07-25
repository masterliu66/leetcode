package function

import (
	"container/heap"
)

type PriorityQueue struct {
	elements   []any
	comparable func(x, y any) bool
}

func NewPriorityQueue(elements func() []any, comparable func(x, y any) bool) *PriorityQueue {
	pq := &PriorityQueue{}
	pq.elements = elements()
	pq.comparable = comparable
	heap.Init(pq)
	return pq
}

func (pq PriorityQueue) Len() int {
	return len(pq.elements)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq.comparable(pq.elements[i], pq.elements[j])
}

func (pq PriorityQueue) Swap(i, j int) {
	pq.elements[i], pq.elements[j] = pq.elements[j], pq.elements[i]
}

func (pq *PriorityQueue) Push(x any) {
	pq.elements = append(pq.elements, x)
}

func (pq *PriorityQueue) Pop() any {
	old := pq.elements
	n := len(old)
	item := old[n-1]
	// avoid memory leak
	old[n-1] = nil
	pq.elements = old[0 : n-1]
	return item
}

func (pq *PriorityQueue) PushElement(element any) {
	heap.Push(pq, element)
}

func (pq *PriorityQueue) PopElement() any {
	if len(pq.elements) > 0 {
		return heap.Pop(pq)
	}
	return nil
}
