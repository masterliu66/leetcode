package function

type IntSet struct {
	*Set
}

func newIntSet() *IntSet {
	return &IntSet{NewSet()}
}

func (set *IntSet) get() int {
	for key := range set.keys {
		return key.(int)
	}
	return 0
}

func (set *IntSet) foreach(consumer func(int)) {
	for key, _ := range set.keys {
		consumer(key.(int))
	}
}