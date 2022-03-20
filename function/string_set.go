package function

type StringSet struct {
	*Set
}

func newStringSet(key string) *StringSet {
	return &StringSet{newSet(key)}
}

func (set *StringSet) get() string {
	for key := range set.keys {
		return key.(string)
	}
	return ""
}