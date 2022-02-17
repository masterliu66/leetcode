package function

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGridIllumination(t *testing.T) {

	n := 5
	lamps := [][]int{{0, 0}, {4, 4}}
	queries := [][]int{{1, 1}, {1, 0}}

	ans := gridIllumination(n, lamps, queries)

	assertEqual(t, []int{1, 0}, ans)
}

func TestCountKDifference(t *testing.T) {

	nums := []int{1, 2, 2, 1}
	k := 1

	ans := countKDifference(nums, k)

	assertEqual(t, 4, ans)
}

func TestSimplifiedFractions(t *testing.T) {

	n := 4

	ans := simplifiedFractions(n)

	assertEqual(t, []string{"1/2", "1/3", "1/4", "2/3", "3/4"}, ans)
}

func TestMinimumDifference(t *testing.T) {

	nums := []int{9, 4, 1, 7}

	k := 3

	ans := minimumDifference(nums, k)

	assertEqual(t, 5, ans)
}

func TestNumEnclaves(t *testing.T) {

	grid := [][]int{{0, 0, 0, 1, 1, 1, 0, 1, 0, 0},
		{0, 1, 1, 0, 0, 0, 1, 0, 1, 0},
		{0, 1, 1, 1, 1, 1, 0, 0, 1, 0},
		{0, 0, 1, 0, 1, 1, 1, 1, 0, 1},
		{0, 1, 1, 0, 0, 0, 1, 1, 1, 1},
		{1, 0, 1, 0, 1, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 1, 0, 0, 0, 1}}

	ans := numEnclaves(grid)

	assertEqual(t, 3, ans)
}

func TestMaxNumberOfBalloons(t *testing.T) {

	text := "loonbalxballpoon"

	ans := maxNumberOfBalloons(text)

	assertEqual(t, 2, ans)
}

func TestSingleNonDuplicate(t *testing.T)  {

	nums := []int{3,3,7,7,10,11,11}

	ans := singleNonDuplicate(nums)

	assertEqual(t, 10, ans)
}

func TestLuckyNumbers(t *testing.T) {

	matrix := [][]int{{1,10,4,2},{9,3,8,7},{15,16,17,12}}

	ans := luckyNumbers(matrix)

	assertEqual(t, []int{12}, ans)
}

func TestKnightProbability(t *testing.T) {

	n, k, row, column := 3, 2, 0, 0

	ans := knightProbability(n, k, row, column)

	assertEqual(t, 0.0625, ans)
}

func assertEqual(t *testing.T, expected interface{}, actual interface{}) {

	a := assert.New(t)

	a.Equal(expected, actual)
}
