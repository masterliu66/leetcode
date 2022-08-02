package function

type CircularQueue struct {
	elements  []int
	rootIndex int
	size      int
	capacity  int
}


func NewCircularQueue(k int) CircularQueue {
	return CircularQueue{make([]int, k), 0, 0, k}
}


func (this *CircularQueue) EnQueue(value int) bool {
	if this.IsFull() {
		return false
	}
	this.size++
	this.elements[this.rearIndex()] = value
	return true
}


func (this *CircularQueue) DeQueue() bool {
	if this.IsEmpty() {
		return false
	}
	this.rootIndex = (this.rootIndex + 1) % this.capacity
	this.size--
	return true
}


func (this *CircularQueue) Front() int {
	if this.IsEmpty() {
		return -1
	}
	return this.elements[this.rootIndex]
}


func (this *CircularQueue) Rear() int {
	if this.IsEmpty() {
		return -1
	}
	return this.elements[this.rearIndex()]
}


func (this *CircularQueue) IsEmpty() bool {
	return this.size == 0
}


func (this *CircularQueue) IsFull() bool {
	return this.size == this.capacity
}


func (this *CircularQueue) rearIndex() int {
	return (this.rootIndex + this.size - 1) % this.capacity
}
