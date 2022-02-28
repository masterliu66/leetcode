package function

import (
	"fmt"
	"math"
	"math/bits"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

/* 1001. 网格照明 */
func gridIllumination(n int, lamps [][]int, queries [][]int) []int {

	type pair struct{ x, y int }

	points := make(map[pair]bool)

	rows := make(map[int]int)
	cols := make(map[int]int)
	diagonals := make(map[int]int)
	antiDiagonals := make(map[int]int)

	/*
	 * (a, b)
	 * 通过灯坐标的行直线与 x 轴的交点, 将交点的 x 坐标作为通过灯坐标的行的数值 => a
	 * 通过灯坐标的列直线与 y 轴的交点, 将交点的 y 坐标作为通过灯坐标的列的数值 => b
	 * 通过灯坐标的正对角线与 x 轴的交点, 将交点的 x 坐标作为通过灯坐标的正对角线的数值 => a - b
	 * 通过灯坐标的反对角线与 y 轴的交点, 将交点的 y 坐标作为通过灯坐标的反对角线的数值 => a + b
	 */
	for _, lamp := range lamps {
		row, col := lamp[0], lamp[1]
		point := pair{row, col}
		if points[point] {
			continue
		}
		points[point] = true
		rows[row]++
		cols[col]++
		diagonals[row-col]++
		antiDiagonals[row+col]++
	}

	ans := make([]int, len(queries))

	for index, query := range queries {
		row, col := query[0], query[1]
		if rows[row] > 0 || cols[col] > 0 || diagonals[row-col] > 0 || antiDiagonals[row+col] > 0 {
			ans[index] = 1
		}
		// 关闭相邻8个方向上的灯
		for i := row - 1; i <= row+1 && i < n; i++ {
			for j := col - 1; j <= col+1 && j < n; j++ {
				if i < 0 || j < 0 {
					continue
				}
				point := pair{i, j}
				if points[point] {
					delete(points, point)
					rows[i]--
					cols[j]--
					diagonals[i-j]--
					antiDiagonals[i+j]--
				}
			}
		}
	}

	return ans
}

/* 2006. 差的绝对值为 K 的数对数目 */
func countKDifference(nums []int, k int) int {

	numMap := map[int]int{}

	ans := 0
	for _, num := range nums {
		ans += numMap[num-k] + numMap[num+k]
		numMap[num]++
	}

	return ans
}

/* 1447. 最简分数 */
func simplifiedFractions(n int) []string {

	var ans []string

	for i := 1; i < n; i++ {
		for j := i + 1; j <= n; j++ {
			if gcd(i, j) == 1 {
				ans = append(ans, strconv.Itoa(i)+"/"+strconv.Itoa(j))
			}
		}
	}

	return ans
}

/* 1984. 学生分数的最小差值 */
func minimumDifference(nums []int, k int) int {

	if k == 1 {
		return 0
	}

	sort.Ints(nums)

	min := math.MaxInt32
	for i, num := range nums[:len(nums)-k+1] {
		min = Min(min, nums[i+k-1]-num)
	}

	return min
}

/* 1020. 飞地的数量 */
func numEnclaves(grid [][]int) int {

	type coordinate struct{ x, y int }

	// 上下左右四个方向
	directions := []coordinate{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}

	m, n := len(grid), len(grid[0])

	// 记录已经访问过的坐标
	vis := map[coordinate]bool{}

	var dfs func(int, int)
	dfs = func(x, y int) {
		current := coordinate{x, y}
		if x < 0 || y < 0 || x >= n || y >= m || grid[y][x] == 0 || vis[current] {
			return
		}
		vis[current] = true
		for _, direction := range directions {
			dfs(x+direction.x, y+direction.y)
		}
	}

	// 从左右边界开始递归搜索
	for i := 0; i < m; i++ {
		dfs(0, i)
		dfs(n-1, i)
	}
	// 从上下边界开始递归搜索
	for i := 1; i < n-1; i++ {
		dfs(i, 0)
		dfs(i, m-1)
	}

	ans := 0
	for y, row := range grid {
		for x, num := range row {
			// 统计没有访问过的陆地坐标数量
			if num == 1 && !vis[coordinate{x, y}] {
				ans++
			}
		}
	}

	return ans
}

/* 1189. “气球” 的最大数量 */
func maxNumberOfBalloons(text string) int {

	balloon := map[rune]int{
		'b': 0,
		'a': 0,
		'l': 0,
		'o': 0,
		'n': 0,
	}
	for _, v := range text {
		_, ok := balloon[v]
		if ok {
			balloon[v]++
		}
	}

	balloon['l'] >>= 1
	balloon['o'] >>= 1
	ans := math.MaxInt32
	for _, ctn := range balloon {
		ans = Min(ans, ctn)
	}

	return ans
}

/* 540. 有序数组中的单一元素 */
func singleNonDuplicate(nums []int) int {

	n := len(nums)

	l, r := 0, n - 1

	for l < r {
		mid := (l + r) >> 1
		// mid为偶数时与右边元素进行比较, 为奇数时与左边元素进行比较, 相同则说明单一元素在mid右侧
		// 当 mid 是偶数时 mid + 1 = mid ^ 1, 当 mid 是奇数时 mid - 1 = mid ^ 1
		if nums[mid] == nums[mid ^ 1] {
			l = mid + 1
		} else {
			r = mid
		}
	}

	return nums[l]
}

/* 1380. 矩阵中的幸运数 */
func luckyNumbers (matrix [][]int) (ans []int) {

	m, n := len(matrix), len(matrix[0])

	minColOfRow := make([]int, m)
	maxOfCol := make([]int, n)

	for i := 0; i < m; i++ {
		min := math.MaxInt32
		for j := 0; j < n; j++ {
			if matrix[i][j] < min {
				min = matrix[i][j]
				minColOfRow[i] = j
			}
			maxOfCol[j] = Max(maxOfCol[j], matrix[i][j])
		}
	}

	for row, col := range minColOfRow {
		if maxOfCol[col] == matrix[row][col] {
			ans = append(ans, matrix[row][col])
		}
	}

	return ans
}

/* 688. 骑士在棋盘上的概率 */
func knightProbability(n int, k int, row int, column int) float64 {

	// 8个可以移动的方向
	direction := []struct{x, y int}{{-2, -1}, {-1, -2}, {1, -2}, {2, -1}, {-2, 1}, {-1, 2}, {1, 2}, {2, 1}}

	// dp[k][i][j] 表示骑士从(i, j)出发, 移动k次后依旧处于棋盘上的概率
	dp := make([][][]float64, k + 1)
	for step := range dp {
		dp[step] = make([][]float64, n)
		for i := 0; i < n; i++ {
			dp[step][i] = make([]float64, n)
			for j := 0; j < n; j++ {
				if step == 0 {
					dp[step][i][j] = 1
				} else {
					for _, d := range direction {
						if x, y := i + d.x, j + d.y; x >= 0 && x < n && y >= 0 && y < n {
							dp[step][i][j] += dp[step-1][x][y] / 8
						}
					}
				}
			}
		}
	}

	return dp[k][row][column]
}

/* #1791 找出星型图的中心节点 */
func findCenter(edges [][]int) int {

	nodes := map[int]int{}
	for _, edge := range edges {
		nodes[edge[0]]++
		nodes[edge[1]]++
	}

	n := len(nodes)
	for node, ctn := range nodes {
		if ctn == n - 1 {
			return node
		}
	}

	panic("没有中心节点")
}

/* 969. 煎饼排序 */
func pancakeSort(arr []int) []int {

	var flip func(int)
	flip = func(k int) {
		for low, high := 0, k; low < high; low, high = low + 1, high - 1 {
			arr[low], arr[high] = arr[high], arr[low]
		}
	}

	var ans []int
	for n := len(arr); n > 1; n-- {
		maxNumIndex := 0
		for i, num := range arr[:n] {
			if num > arr[maxNumIndex] {
				maxNumIndex = i
			}
		}
		if maxNumIndex == n - 1  {
			continue
		}
		if maxNumIndex != 0 {
			// 将最大数移至首位
			flip(maxNumIndex)
			ans = append(ans, maxNumIndex + 1)
		}
		// 翻转前right个数
		flip(n - 1)
		ans = append(ans, n)
	}

	return ans
}

/* 717. 1比特与2比特字符 */
func isOneBitCharacter(bits []int) bool {

	n, i := len(bits), len(bits) - 2
	for i >= 0 && bits[i] == 1 {
		i--
	}

	return (n - i) % 2 == 0
}

/* 838. 推多米诺 */
func pushDominoes(dominoes string) string {

	ans := []byte(dominoes)
	i, n, left := 0, len(dominoes), byte('L')
	for i < n {
		j, right := i, byte('R')
		// 找到一段连续的没有被推动的骨牌
		for j < n && ans[j] == '.' {
			j++
		}
		if j < n {
			right = ans[j]
		}
		// 方向相同
		if left == right {
			for ; i < j; i++ {
				ans[i] = right
			}
		// 方向相反
		} else if left == 'R' && right == 'L' {
			for k := j - 1; i < k; k-- {
				ans[i] = 'R'
				ans[k] = 'L'
				i++
			}
		}
		// 记录左边多米诺骨牌的方向
		left = right
		i = j + 1
	}

	return string(ans)
}

/* 1994. 好子集的数目 */
func numberOfGoodSubsets(nums []int) int {

	const mod int = 1e9 + 7

	var ctn [30]int
	for _, num := range nums {
		ctn[num - 1]++
	}

	primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
	masks := map[int]int {
		// 质数
		2: 0x01, 3: 0x02, 5: 0x04, 7: 0x08, 11: 0x10, 13: 0x20, 17: 0x40, 19: 0x80, 23: 0x100, 29: 0x200,
		// 可以分解为不同质因数的合数
		6: 0x03, 10: 0x05, 14: 0x09, 15: 0x06, 21: 0x0a, 22: 0x11, 26: 0x21, 30: 0x07,
	}

	// dp[i][mask] 表示只取<=i的数, 在状态mask下的方案数
	dp := make([]int, 1 << len(primes))
	dp[0] = 1
	// 所有数字1都可以取或者不取
	for i := 0; i < ctn[0]; i++ {
		dp[0] = dp[0] * 2 % mod
	}

	for i := 2; i <= 30; i++ {
		if ctn[i - 1] == 0 || masks[i] == 0 {
			continue
		}
		for mask := 1 << len(primes); mask > 0; mask-- {
			// masks[i]为当前状态的子集
			if mask & masks[i] == masks[i] {
				dp[mask] = (dp[mask] + dp[mask ^ (masks[i])] * ctn[i - 1]) % mod
			}
		}
	}

	ans := 0
	for _, v := range dp[1:] {
		ans = (ans + v) % mod
	}

	return ans
}

/* 917. 仅仅反转字母 */
func reverseOnlyLetters(s string) string {

	n := len(s)

	runes := []rune(s)

	for l, r := 0, n - 1; l < r; {
		if !unicode.IsLetter(runes[l]) {
			l++
			continue
		}
		if !unicode.IsLetter(runes[r]) {
			r--
			continue
		}
		runes[l], runes[r] = runes[r], runes[l]
		l++
		r--
	}

	return string(runes)
}

/* 1706. 球会落何处 */
func findBall(grid [][]int) []int {

	m, n := len(grid), len(grid[0])

	var dfs func(int, int) int
	dfs = func(row, col int) int {

		if row == m {
			return col
		}

		direction := grid[row][col]
		col += direction
		if col < 0 || col == n || grid[row][col] != direction {
			return -1
		}

		return dfs(row + 1, col)
	}

	ans := make([]int, n)
	for i := 0; i < n; i++ {
		ans[i] = dfs(0, i)
	}

	return ans
}

/* 537. 复数乘法 */
func complexNumberMultiply(num1 string, num2 string) string {

	num1Complex, _ := strconv.ParseComplex(num1, 128)
	num2Complex, _ := strconv.ParseComplex(num2, 128)

	ansComplex := num1Complex * num2Complex

	return fmt.Sprintf("%d+%di", int (real(ansComplex)), int (imag(ansComplex)))
}

func Equal(a, b []int) bool {

	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if b[i] != v {
			return false
		}
	}

	return true
}

/* 2016. 增量元素之间的最大差值 */
func maximumDifference(nums []int) int {

	// 记录最小值
	min := nums[0]

	// 记录最大差值
	maxDiff := -1
	for _, num := range nums[1:] {
		if num > min {
			maxDiff = Max(maxDiff, num - min)
		} else {
			min = num
		}
	}

	return maxDiff
}

/* 553. 最优除法 */
func optimalDivision(nums []int) string {

	n := len(nums)
	if n == 1 {
		return strconv.Itoa(nums[0])
	}

	if n == 2 {
		return fmt.Sprintf("%d/%d", nums[0], nums[1])
	}

	var builder strings.Builder
	builder.WriteString(strconv.Itoa(nums[0]))
	builder.WriteString("/(")
	builder.WriteString(strconv.Itoa(nums[1]))
	for _, num := range nums[2:] {
		builder.WriteString("/")
		builder.WriteString(strconv.Itoa(num))
	}
	builder.WriteString(")")

	return builder.String()
}

/* 1601. 最多可达成的换楼请求数目 */
func maximumRequests(n int, requests [][]int) (ans int) {

	m := len(requests)

	outer: for mask := 1; mask < 1 << m; mask++ {
		onesCount := bits.OnesCount(uint(mask))
		if onesCount <= ans {
			continue
		}
		ctn := make([]int, n)
		for i, request := range requests {
			if (mask >> i) & 1 == 1 {
				ctn[request[0]]--
				ctn[request[1]]++
			}
		}
		for _, v := range ctn {
			if v != 0 {
				continue outer
			}
		}
		ans = onesCount
	}

	return
}

func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func Max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

/* 求两个数的最大公约数 */
func gcd(a, b int) int {

	for b != 0 {
		a, b = b, a%b
	}

	return a
}
