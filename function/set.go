package function

type Set struct {
	keys map[string]struct{}
}

func newSet(key string) Set {
	return Set{map[string]struct{}{key: {}}}
}

func (set *Set) len() int {
	return len(set.keys)
}

func (set *Set) isEmpty() bool {
	return set.len() == 0
}

func (set *Set) contains(key string) bool {
	_, ok := set.keys[key]
	return ok
}

func (set *Set) get() string {
	for key := range set.keys {
		return key
	}
	return ""
}

func (set *Set) add(key string) {
	set.keys[key] = struct{}{}
}

func (set *Set) remove(key string) {
	delete(set.keys, key)
}

func (set *Set) clear() {
	set.keys = map[string]struct{}{}
}

