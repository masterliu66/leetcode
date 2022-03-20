package function

type Graph struct {
	vertexes *IntSet
	edges [][]int
	next [][]int
}

func NewGraph(edges [][]int) *Graph {

	set := newIntSet()

	for _, edge := range edges {
		set.add(edge[0])
		set.add(edge[1])
	}

	next := make([][]int, set.len())
	for _, edge := range edges {
		next[edge[0]] = append(next[edge[0]], edge[1])
		next[edge[1]] = append(next[edge[1]], edge[0])
	}

	return &Graph{set, edges, next}
}