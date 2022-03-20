package function

type Set struct {
	keys map[interface{}]struct{}
}

func NewSet() *Set {
	return &Set{map[interface{}]struct{}{}}
}

func newSet(key interface{}) *Set {
	return &Set{map[interface{}]struct{}{key: {}}}
}

func (set *Set) len() int {
	return len(set.keys)
}

func (set *Set) isEmpty() bool {
	return set.len() == 0
}

func (set *Set) contains(key interface{}) bool {
	_, ok := set.keys[key]
	return ok
}

func (set *Set) get() interface{} {
	for key := range set.keys {
		return key
	}
	return ""
}

func (set *Set) add(key interface{}) {
	set.keys[key] = struct{}{}
}

func (set *Set) remove(key interface{}) {
	delete(set.keys, key)
}

func (set *Set) clear() {
	set.keys = map[interface{}]struct{}{}
}

