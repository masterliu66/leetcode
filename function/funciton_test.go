package function

import (
	"fmt"
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

func TestSingleNonDuplicate(t *testing.T) {

	nums := []int{3, 3, 7, 7, 10, 11, 11}

	ans := singleNonDuplicate(nums)

	assertEqual(t, 10, ans)
}

func TestLuckyNumbers(t *testing.T) {

	matrix := [][]int{{1, 10, 4, 2}, {9, 3, 8, 7}, {15, 16, 17, 12}}

	ans := luckyNumbers(matrix)

	assertEqual(t, []int{12}, ans)
}

func TestKnightProbability(t *testing.T) {

	n, k, row, column := 3, 2, 0, 0

	ans := knightProbability(n, k, row, column)

	assertEqual(t, 0.0625, ans)
}

func TestFindCenter(t *testing.T) {

	edges := [][]int{{1, 2}, {5, 1}, {1, 3}, {1, 4}}

	ans := findCenter(edges)

	assertEqual(t, 1, ans)
}

func TestPancakeSort(t *testing.T) {

	arr := []int{3, 2, 4, 1}

	ans := pancakeSort(arr)

	assertEqual(t, []int{1, 2, 3, 4}, arr)
	assertLessOrEqual(t, len(ans), len(arr)*10)
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

	nums := []int{4, 2, 3, 15}

	ans := numberOfGoodSubsets(nums)

	assertEqual(t, 5, ans)
}

func TestReverseOnlyLetters(t *testing.T) {

	s := "Test1ng-Leet=code-Q!"

	ans := reverseOnlyLetters(s)

	assertEqual(t, "Qedo1ct-eeLg=ntse-T!", ans)
}

func TestFindBall(t *testing.T) {

	grid := [][]int{{1, 1, 1, 1, 1, 1},
		{-1, -1, -1, -1, -1, -1},
		{1, 1, 1, 1, 1, 1},
		{-1, -1, -1, -1, -1, -1}}

	ans := findBall(grid)

	assertEqual(t, []int{0, 1, 2, 3, 4, -1}, ans)
}

func TestComplexNumberMultiply(t *testing.T) {

	num1, num2 := "1+-1i", "1+-1i"

	ans := complexNumberMultiply(num1, num2)

	assertEqual(t, "0+-2i", ans)
}

func TestMaximumDifference(t *testing.T) {

	nums := []int{7, 1, 5, 4}

	ans := maximumDifference(nums)

	assertEqual(t, 4, ans)
}

func TestOptimalDivision(t *testing.T) {

	nums := []int{1000, 100, 10, 2}

	ans := optimalDivision(nums)

	assertEqual(t, "1000/(100/10/2)", ans)
}

func TestMaximumRequests(t *testing.T) {

	n := 5
	requests := [][]int{{0, 1}, {1, 0}, {0, 1}, {1, 2}, {2, 0}, {3, 4}}

	ans := maximumRequests(n, requests)

	assertEqual(t, 5, ans)
}

func TestConvert(t *testing.T) {

	s := "PAYPALISHIRING"
	numRows := 3

	ans := convert(s, numRows)

	assertEqual(t, "PAHNAPLSIIGYIR", ans)
}

func TestSubArrayRanges(t *testing.T) {

	nums := []int{4, -2, -3, 4, 1}

	ans := subArrayRanges(nums)

	assertEqual(t, int64(59), ans)
}

func TestFindLUSlength(t *testing.T) {

	a, b := "aba", "cdc"

	ans := findLUSlength(a, b)

	assertEqual(t, 3, ans)
}

func TestGoodDaysToRobBank(t *testing.T) {

	security := []int{5, 3, 3, 3, 5, 6, 2}
	time := 2

	ans := goodDaysToRobBank(security, time)

	assertEqual(t, []int{2, 3}, ans)
}

func TestConvertToBase7(t *testing.T) {

	num := -100

	ans := convertToBase7(num)

	assertEqual(t, "-202", ans)
}

func TestPlatesBetweenCandles(t *testing.T) {

	s := "***|**|*****|**||**|*"

	queries := [][]int{{1, 17}, {4, 5}, {14, 17}, {5, 11}, {15, 16}}

	ans := platesBetweenCandles(s, queries)

	assertEqual(t, []int{9, 0, 0, 0, 0}, ans)
}

func TestBestRotation(t *testing.T) {

	nums := []int{2, 3, 1, 4, 0}

	ans := bestRotation(nums)

	assertEqual(t, 3, ans)
}

func TestPreorder(t *testing.T) {

	nums := []int{1, NULL, 3, 2, 4, NULL, 5, 6}

	root := NewTree(nums)

	ans := preorder(root)

	assertEqual(t, []int{1, 3, 5, 6, 2, 4}, ans)
}

func TestCountHighestScoreNodes(t *testing.T) {

	parents := []int{-1, 2, 0, 2, 0}

	ans := countHighestScoreNodes(parents)

	assertEqual(t, 3, ans)
}

func TestPostorder(t *testing.T) {

	nums := []int{1, NULL, 3, 2, 4, NULL, 5, 6}

	root := NewTree(nums)

	ans := postorder(root)

	assertEqual(t, []int{5, 6, 3, 2, 4, 1}, ans)
}

func TestValidUtf8(t *testing.T) {

	data := []int{235, 140, 4}

	ans := validUtf8(data)

	assertEqual(t, false, ans)
}

func TestFindRestaurant(t *testing.T) {

	list1 := []string{"Shogun", "Tapioca Express", "Burger King", "KFC"}
	list2 := []string{"KFC", "Burger King", "Tapioca Express", "Shogun"}

	ans := findRestaurant(list1, list2)

	assertEqual(t, []string{"KFC", "Burger King", "Tapioca Express", "Shogun"}, ans)
}

func TestCountMaxOrSubsets(t *testing.T) {

	nums := []int{3, 2, 1, 5}

	ans := countMaxOrSubsets(nums)

	assertEqual(t, 6, ans)
}

func TestAllOne(t *testing.T) {

	allOne := Constructor()
	allOne.Inc("hello")
	allOne.Inc("hello")
	assertEqual(t, "hello", allOne.GetMaxKey())
	assertEqual(t, "hello", allOne.GetMinKey())
	allOne.Inc("leet")
	assertEqual(t, "hello", allOne.GetMaxKey())
	assertEqual(t, "leet", allOne.GetMinKey())
}

func TestTree2str(t *testing.T) {

	nums := []int{1, 2, 3, NULL, 4}

	root := NewBinaryTree(nums)

	ans := tree2str(root)

	assertEqual(t, "1(2()(4))(3)", ans)
}

func TestNetworkBecomesIdle(t *testing.T) {

	edges := [][]int{{0, 1}, {0, 2}, {1, 2}}
	patience := []int{0, 2, 1}

	ans := networkBecomesIdle(edges, patience)

	assertEqual(t, 4, ans)
}

func TestFindTarget(t *testing.T) {

	nums := []int{5, 3, 6, 2, 4, NULL, 7}
	k := 28

	root := NewBinaryTree(nums)

	ans := findTarget(root, k)

	assertEqual(t, false, ans)
}

func TestWinnerOfGame(t *testing.T) {

	colors := "AAABABB"

	ans := winnerOfGame(colors)

	assertEqual(t, true, ans)
}

func TestMaxConsecutiveAnswers(t *testing.T) {

	answerKey, k := "TTFTTFTT", 1

	ans := maxConsecutiveAnswers(answerKey, k)

	assertEqual(t, 5, ans)
}

func TestRotateString(t *testing.T) {

	s, goal := "abcde", "cdeab"

	ans := rotateString(s, goal)

	assertEqual(t, true, ans)
}

func TestLevelOrder(t *testing.T) {

	nums := []int{1, NULL, 2, 3, 4, 5, NULL, NULL, 6, 7, NULL, 8, NULL, 9, 10, NULL, NULL, 11, NULL, 12, NULL, 13, NULL, NULL, 14}

	root := NewTree(nums)

	ans := levelOrder(root)

	assertEqual(t, [][]int{{1}, {2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13}, {14}}, ans)
}

func TestCountNumbersWithUniqueDigits(t *testing.T) {

	nums := []int{1, 10, 91, 739, 5275, 32491, 168571, 712891, 2345851}

	for i, num := range nums {
		ans := countNumbersWithUniqueDigits(i)
		assertEqual(t, num, ans)
	}
}

func TestConstruct(t *testing.T) {

	grid := [][]int{{1, 1, 1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0, 0, 0}}

	ans := construct(grid)

	fmt.Println(ans)
}

func TestNumSubarrayProductLessThanK(t *testing.T) {

	nums := []int{10, 5, 2, 6}
	k := 100

	ans := numSubarrayProductLessThanK(nums, k)

	assertEqual(t, 8, ans)
}

func TestCircularQueue(t *testing.T) {

	k := 3
	circularQueue := NewCircularQueue(k)
	assertEqual(t, true, circularQueue.EnQueue(1))
	assertEqual(t, true, circularQueue.EnQueue(2))
	assertEqual(t, true, circularQueue.EnQueue(3))
	assertEqual(t, false, circularQueue.EnQueue(4))
	assertEqual(t, 3, circularQueue.Rear())
	assertEqual(t, true, circularQueue.IsFull())
	assertEqual(t, true, circularQueue.DeQueue())
	assertEqual(t, true, circularQueue.EnQueue(4))
	assertEqual(t, 4, circularQueue.Rear())
	assertEqual(t, 2, circularQueue.Front())
}

func TestMinSubsequence(t *testing.T) {
	nums := []int{4, 3, 10, 9, 8}
	ans := minSubsequence(nums)
	assertEqual(t, []int{10, 9}, ans)
}
