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

func TestFindCenter(t *testing.T) {

	edges := [][]int{{1,2},{5,1},{1,3},{1,4}}

	ans := findCenter(edges)

	assertEqual(t, 1, ans)
}

func TestPancakeSort(t *testing.T) {

	arr := []int{3,2,4,1}

	ans := pancakeSort(arr)

	assertEqual(t, []int{1,2,3,4}, arr)
	assertLessOrEqual(t, len(ans), len(arr) * 10)
}

func TestIsOneBitCharacter(t *testing.T) {

	bits := []int{1, 1, 1, 0}

	ans := isOneBitCharacter(bits)

	assertEqual(t, false, ans)
}

func TestPushDominoes(t *testing.T) {

	dominoes := ".L.R...LR..L.."

	ans := pushDominoes(dominoes)

	assertEqual(t, "LL.RR.LLRRLL..", ans)
}

func TestNumberOfGoodSubsets(t *testing.T) {

	nums := []int{4,2,3,15}

	ans := numberOfGoodSubsets(nums)

	assertEqual(t, 5, ans)
}

func TestReverseOnlyLetters(t *testing.T) {

	s := "Test1ng-Leet=code-Q!"

	ans := reverseOnlyLetters(s)

	assertEqual(t, "Qedo1ct-eeLg=ntse-T!", ans)
}

func assertEqual(t *testing.T, expected interface{}, actual interface{}) {

	a := assert.New(t)

	a.Equal(expected, actual)
}

func assertLessOrEqual(t *testing.T, e1 interface{}, e2 interface{}) {

	a := assert.New(t)

	a.LessOrEqual(e1, e2)
}
