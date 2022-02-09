package function

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGridIllumination(t *testing.T) {

	n := 5
	lamps := [][]int{{0, 0}, {4, 4}}
	queries := [][]int{{1, 1}, {1, 0}}

	gridIllumination(n, lamps, queries)
}

func TestCountKDifference(t *testing.T) {

	nums := []int{1, 2, 2, 1}
	k := 1

	ans := countKDifference(nums, k)

	a := assert.New(t)
	a.Equal(ans, 4)
}
