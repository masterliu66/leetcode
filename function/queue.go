package function

type Queue struct {
	elements []interface{}
}

func (queue *Queue) len() int {
	return len(queue.elements)
}

func (queue *Queue) isEmpty() bool {
	return queue.len() == 0
}

func (queue *Queue) peek() interface{} {
	return queue.elements[queue.len()-1]
}

func (queue *Queue) poll() interface{} {
	if queue.isEmpty() {
		return nil
	}
	element := queue.elements[0]
	queue.elements = queue.elements[1:]
	return element
}

func (queue *Queue) offer(element interface{}) {
	queue.elements = append(queue.elements, element)
}

func (queue *Queue) clear() {
	queue.elements = queue.elements[:0]
}
