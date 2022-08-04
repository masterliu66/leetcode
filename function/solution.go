package function

import (
	"container/list"
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
			if Gcd(i, j) == 1 {
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

	l, r := 0, n-1

	for l < r {
		mid := (l + r) >> 1
		// mid为偶数时与右边元素进行比较, 为奇数时与左边元素进行比较, 相同则说明单一元素在mid右侧
		// 当 mid 是偶数时 mid + 1 = mid ^ 1, 当 mid 是奇数时 mid - 1 = mid ^ 1
		if nums[mid] == nums[mid^1] {
			l = mid + 1
		} else {
			r = mid
		}
	}

	return nums[l]
}

/* 1380. 矩阵中的幸运数 */
func luckyNumbers(matrix [][]int) (ans []int) {

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
	direction := []struct{ x, y int }{{-2, -1}, {-1, -2}, {1, -2}, {2, -1}, {-2, 1}, {-1, 2}, {1, 2}, {2, 1}}

	// dp[k][i][j] 表示骑士从(i, j)出发, 移动k次后依旧处于棋盘上的概率
	dp := make([][][]float64, k+1)
	for step := range dp {
		dp[step] = make([][]float64, n)
		for i := 0; i < n; i++ {
			dp[step][i] = make([]float64, n)
			for j := 0; j < n; j++ {
				if step == 0 {
					dp[step][i][j] = 1
				} else {
					for _, d := range direction {
						if x, y := i+d.x, j+d.y; x >= 0 && x < n && y >= 0 && y < n {
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
		if ctn == n-1 {
			return node
		}
	}

	panic("没有中心节点")
}

/* 969. 煎饼排序 */
func pancakeSort(arr []int) []int {

	var flip func(int)
	flip = func(k int) {
		for low, high := 0, k; low < high; low, high = low+1, high-1 {
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
		if maxNumIndex == n-1 {
			continue
		}
		if maxNumIndex != 0 {
			// 将最大数移至首位
			flip(maxNumIndex)
			ans = append(ans, maxNumIndex+1)
		}
		// 翻转前right个数
		flip(n - 1)
		ans = append(ans, n)
	}

	return ans
}

/* 717. 1比特与2比特字符 */
func isOneBitCharacter(bits []int) bool {

	n, i := len(bits), len(bits)-2
	for i >= 0 && bits[i] == 1 {
		i--
	}

	return (n-i)%2 == 0
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
		ctn[num-1]++
	}

	primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
	masks := map[int]int{
		// 质数
		2: 0x01, 3: 0x02, 5: 0x04, 7: 0x08, 11: 0x10, 13: 0x20, 17: 0x40, 19: 0x80, 23: 0x100, 29: 0x200,
		// 可以分解为不同质因数的合数
		6: 0x03, 10: 0x05, 14: 0x09, 15: 0x06, 21: 0x0a, 22: 0x11, 26: 0x21, 30: 0x07,
	}

	// dp[i][mask] 表示只取<=i的数, 在状态mask下的方案数
	dp := make([]int, 1<<len(primes))
	dp[0] = 1
	// 所有数字1都可以取或者不取
	for i := 0; i < ctn[0]; i++ {
		dp[0] = dp[0] * 2 % mod
	}

	for i := 2; i <= 30; i++ {
		if ctn[i-1] == 0 || masks[i] == 0 {
			continue
		}
		for mask := 1 << len(primes); mask > 0; mask-- {
			// masks[i]为当前状态的子集
			if mask&masks[i] == masks[i] {
				dp[mask] = (dp[mask] + dp[mask^(masks[i])]*ctn[i-1]) % mod
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

	for l, r := 0, n-1; l < r; {
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

		return dfs(row+1, col)
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

	return fmt.Sprintf("%d+%di", int(real(ansComplex)), int(imag(ansComplex)))
}

/* 2016. 增量元素之间的最大差值 */
func maximumDifference(nums []int) int {

	// 记录最小值
	min := nums[0]

	// 记录最大差值
	maxDiff := -1
	for _, num := range nums[1:] {
		if num > min {
			maxDiff = Max(maxDiff, num-min)
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

outer:
	for mask := 1; mask < 1<<m; mask++ {
		onesCount := bits.OnesCount(uint(mask))
		if onesCount <= ans {
			continue
		}
		ctn := make([]int, n)
		for i, request := range requests {
			if (mask>>i)&1 == 1 {
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

/* 6. Z 字形变换 */
func convert(s string, numRows int) string {

	n := len(s)
	if numRows == 1 || numRows >= n {
		return s
	}
	t := numRows*2 - 2
	ans := make([]byte, 0, n)
	// 枚举矩阵的行
	for i := 0; i < numRows; i++ {
		// 枚举每个周期的起始下标
		for j := 0; j+i < n; j += t {
			// 当前周期的第一个字符
			ans = append(ans, s[j+i])
			// 首行和末行只有一个字符
			if i == 0 || i == numRows-1 {
				continue
			}
			if j+t-i < n {
				// 当前周期的第二个字符
				ans = append(ans, s[j+t-i])
			}
		}
	}
	return string(ans)
}

/* 2104. 子数组范围和 */
func subArrayRanges(nums []int) int64 {

	type Tuple struct{ minLeft, maxLeft, minRight, maxRight int }

	n := len(nums)

	f := make([]Tuple, n)
	var minStack, maxStack Stack
	for i, num := range nums {
		for !minStack.isEmpty() && nums[minStack.peek()] > num {
			minStack.pop()
		}
		if minStack.isEmpty() {
			f[i].minLeft = -1
		} else {
			f[i].minLeft = minStack.peek()
		}
		minStack.push(i)
		// nums[maxStack.peek()] == nums[i]时maxStack.peek() < i
		for !maxStack.isEmpty() && nums[maxStack.peek()] <= num {
			maxStack.pop()
		}
		if maxStack.isEmpty() {
			f[i].maxLeft = -1
		} else {
			f[i].maxLeft = maxStack.peek()
		}
		maxStack.push(i)
	}

	minStack.clear()
	maxStack.clear()
	for i := n - 1; i >= 0; i-- {
		num := nums[i]
		// nums[minStack.peek()] == nums[i]时minStack.peek() > i
		for !minStack.isEmpty() && nums[minStack.peek()] >= num {
			minStack.pop()
		}
		if minStack.isEmpty() {
			f[i].minRight = n
		} else {
			f[i].minRight = minStack.peek()
		}
		minStack.push(i)
		for !maxStack.isEmpty() && nums[maxStack.peek()] < num {
			maxStack.pop()
		}
		if maxStack.isEmpty() {
			f[i].maxRight = n
		} else {
			f[i].maxRight = maxStack.peek()
		}
		maxStack.push(i)
	}

	var sumMax, sumMin int64
	for i, num := range nums {
		sumMax += int64(f[i].maxRight-i) * int64(i-f[i].maxLeft) * int64(num)
		sumMin += int64(f[i].minRight-i) * int64(i-f[i].minLeft) * int64(num)
	}

	return sumMax - sumMin
}

/* 521. 最长特殊序列 Ⅰ */
func findLUSlength(a string, b string) int {

	if a == b {
		return -1
	}

	return Max(len(a), len(b))
}

/* 2100. 适合打劫银行的日子 */
func goodDaysToRobBank(security []int, time int) []int {

	n := len(security)
	// f[i]表示第i天前连续非递增的最大天数
	f := make([]int, n)
	// g[i]表示第i天后连续非递减的最大天数
	g := make([]int, n)
	for i := 1; i < n; i++ {
		if security[i] <= security[i-1] {
			f[i] = f[i-1] + 1
		}
		if security[n-i-1] <= security[n-i] {
			g[n-i-1] = g[n-i] + 1
		}
	}

	var ans []int
	for i := time; i < n-time; i++ {
		if f[i] >= time && g[i] >= time {
			ans = append(ans, i)
		}
	}

	return ans
}

/* 504. 七进制数 */
func convertToBase7(num int) string {

	var digits []rune
	flag := false
	if num < 0 {
		num = -num
		flag = true
	}

	for num >= 7 {
		digits = append(digits, '0'+rune(num%7))
		num /= 7
	}

	digits = append(digits, '0'+rune(num))

	if flag {
		digits = append(digits, '-')
	}

	for i, n := 0, len(digits); i < n/2; i++ {
		digits[i], digits[n-1-i] = digits[n-1-i], digits[i]
	}

	return string(digits)
}

/* 2055. 蜡烛之间的盘子 */
func platesBetweenCandles(s string, queries [][]int) []int {

	n := len(s)

	plateCtn := make([]int, n+1)
	candleCtn := make([]int, n+1)
	// k,v表示蜡烛数量为k时的最小下标v
	candleMinIndex := map[int]int{}
	candleMaxIndex := map[int]int{}
	for i, v := range s {
		if v == '*' {
			plateCtn[i+1] = plateCtn[i] + 1
			candleCtn[i+1] = candleCtn[i]
			candleMaxIndex[candleCtn[i]] = i
		} else {
			plateCtn[i+1] = plateCtn[i]
			candleCtn[i+1] = candleCtn[i] + 1
			candleMinIndex[candleCtn[i+1]] = i
		}
	}

	ans := make([]int, len(queries))
	for i, query := range queries {
		left := query[0]
		right := query[1]
		// 两只蜡烛以下, 不可能有盘子处于蜡烛之间
		if candleCtn[right+1]-candleCtn[left] <= 1 {
			ans[i] = 0
			continue
		}
		ans[i] = plateCtn[right+1] - plateCtn[left]
		// 查找与最左边盘子数相同的最大下标
		if s[left] == '*' && candleMaxIndex[candleCtn[left+1]] >= left {
			ans[i] -= candleMaxIndex[candleCtn[left+1]] - left + 1
		}
		// 查找与最右边盘子数相同的最小下标
		if s[right] == '*' && candleMinIndex[candleCtn[right+1]] > 0 && candleMinIndex[candleCtn[right+1]] <= right {
			ans[i] -= right - candleMinIndex[candleCtn[right+1]]
		}
	}

	return ans
}

/* 798. 得分最高的最小轮调 */
func bestRotation(nums []int) int {

	n := len(nums)

	// f[i]表示k=i时的得分数
	f := make([]int, n)
	// g[i]表示k=i时数组值等于下标的数量
	g := make([]int, n)

	for i, num := range nums {
		g[(i+n-num)%n]++
		if num <= i {
			f[0]++
		}
	}

	ans := 0
	for i := 1; i < n; i++ {
		// 每次移动nums[1:n)下标-1, nums[0]下标变为n-1
		// 数组值小于下标的依旧可得分, 数组值等于下标的扣分, 数组值大于下标的依旧不得分, 下标为0的必定可加分
		f[i] = f[i-1] - g[i-1] + 1
		if f[i] > f[ans] {
			ans = i
		}
	}

	return ans
}

/* 589. N 叉树的前序遍历 */
func preorder(root *Node) []int {

	var ans []int

	var dfs func(*Node)
	dfs = func(node *Node) {
		if node == nil {
			return
		}

		ans = append(ans, node.Val)

		for _, child := range node.Children {
			dfs(child)
		}
	}

	dfs(root)

	return ans
}

/* 2049. 统计最高分的节点数目 */
func countHighestScoreNodes(parents []int) int {

	n := len(parents)

	// children[i]表示节点i的孩子节点
	children := make([][]int, n)
	// degree[i]表示节点i的度
	degree := make([]int, n)

	for i := 1; i < n; i++ {
		parent := parents[i]
		children[parent] = append(children[parent], i)
	}

	var dfs func(int) int
	dfs = func(current int) int {
		if len(children[current]) == 0 {
			degree[current] = 1
			return 1
		}
		if degree[current] != 0 {
			return degree[current]
		}
		sum := 0
		for _, i := range children[current] {
			sum += dfs(i)
		}
		degree[current] = sum + 1
		return degree[current]
	}

	dfs(0)

	max := 0
	ans := 0
	for i := 0; i < n; i++ {
		// 父节点分数 = 跟节点的度 - 当前节点的度
		score := degree[0] - degree[i]
		if len(children[i]) > 0 {
			// 子节点分数 = 子节点的度
			for _, j := range children[i] {
				if score == 0 {
					score = degree[j]
				} else {
					score *= degree[j]
				}
			}
		}
		// 找出最大分数
		if score == max {
			ans++
		} else if score > max {
			max = score
			ans = 1
		}
	}

	return ans
}

/* 590. N 叉树的后序遍历 */
func postorder(root *Node) []int {

	var ans []int

	var dfs func(*Node)
	dfs = func(node *Node) {
		if node == nil {
			return
		}

		for _, child := range node.Children {
			dfs(child)
		}

		ans = append(ans, node.Val)
	}

	dfs(root)

	return ans
}

/* 393. UTF-8 编码验证 */
func validUtf8(data []int) bool {

	n := len(data)
	for i := 0; i < n; i++ {
		// n 字节 的字符
		if (data[i]>>7)&1 == 1 {
			ctn := 0
			for bit := 6; bit >= 0; bit-- {
				if (data[i]>>bit)&1 == 0 {
					break
				}
				ctn++
			}
			// 字节数为1或字节数超过4个或剩余字节数不足
			if ctn == 0 || ctn > 3 || i+ctn >= n {
				return false
			}
			for j := 0; j < ctn; j++ {
				i++
				// 不是10开头
				if (data[i]>>7)&1 == 0 || (data[i]>>6)&1 == 1 {
					return false
				}
			}
		}
		// 1 字节 的字符可直接跳过
	}

	return true
}

/* 599. 两个列表的最小索引总和 */
func findRestaurant(list1 []string, list2 []string) (ans []string) {

	len1, len2 := len(list1), len(list2)

	indexes := map[string]int{}
	var shorterList, longerList []string
	if len1 <= len2 {
		shorterList = list1
		longerList = list2
	} else {
		shorterList = list2
		longerList = list1
	}

	for i, s := range shorterList {
		indexes[s] = i
	}

	min := math.MaxInt32
	for i, s := range longerList {
		if j, exist := indexes[s]; exist {
			if i+j < min {
				min = i + j
				ans = []string{s}
			} else if i+j == min {
				ans = append(ans, s)
			}
		}
	}

	return
}

/* 2044. 统计按位或能得到最大值的子集数目 */
func countMaxOrSubsets(nums []int) int {

	n := len(nums)

	max := 0
	ans := 0
	for mask := 1; mask < 1<<n; mask++ {
		val := 0
		for i, num := range nums {
			if mask>>i&1 == 1 {
				val |= num
			}
		}
		if val > max {
			max = val
			ans = 1
		} else if val == max {
			ans++
		}
	}

	return ans
}

/* 432. 全 O(1) 的数据结构 */
type AllOne struct {
	*list.List
	nodes map[string]*list.Element
}

type AllOneNode struct {
	keys  *StringSet
	count int
}

func Constructor() AllOne {
	return AllOne{list.New(), map[string]*list.Element{}}
}

func (this *AllOne) Inc(key string) {
	node, ok := this.nodes[key]
	if ok {
		curNode := node.Value.(AllOneNode)
		if next := node.Next(); next == nil || next.Value.(AllOneNode).count > curNode.count+1 {
			this.nodes[key] = this.InsertAfter(AllOneNode{newStringSet(key), curNode.count + 1}, node)
		} else {
			keys := next.Value.(AllOneNode).keys
			keys.add(key)
			this.nodes[key] = next
		}
		curNode.keys.remove(key)
		if curNode.keys.isEmpty() {
			this.Remove(node)
		}
	} else {
		if this.Front() == nil || this.Front().Value.(AllOneNode).count > 1 {
			this.nodes[key] = this.PushFront(AllOneNode{newStringSet(key), 1})
		} else {
			keys := this.Front().Value.(AllOneNode).keys
			keys.add(key)
			this.nodes[key] = this.Front()
		}
	}
}

func (this *AllOne) Dec(key string) {
	cur := this.nodes[key]
	curNode := cur.Value.(AllOneNode)
	if curNode.count > 1 {
		if pre := cur.Prev(); pre == nil || pre.Value.(AllOneNode).count < curNode.count-1 {
			this.nodes[key] = this.InsertBefore(AllOneNode{newStringSet(key), curNode.count - 1}, cur)
		} else {
			keys := pre.Value.(AllOneNode).keys
			keys.add(key)
			this.nodes[key] = pre
		}
	} else { // key 仅出现一次，将其移出 nodes
		delete(this.nodes, key)
	}
	curNode.keys.remove(key)
	if curNode.keys.isEmpty() {
		this.Remove(cur)
	}
}

func (this *AllOne) GetMaxKey() string {
	element := this.Back()
	if element != nil {
		keys := element.Value.(AllOneNode).keys
		return keys.get()
	}
	return ""
}

func (this *AllOne) GetMinKey() string {
	element := this.Front()
	if element != nil {
		keys := element.Value.(AllOneNode).keys
		return keys.get()
	}
	return ""
}

/* 606. 根据二叉树创建字符串 */
func tree2str(root *TreeNode) string {

	builder := strings.Builder{}

	var preOrder func(*TreeNode)
	preOrder = func(node *TreeNode) {

		if node == nil {
			return
		}

		if node != root {
			builder.WriteString("(")
		}

		builder.WriteString(strconv.Itoa(node.Val))

		if node.Left != nil {
			preOrder(node.Left)

		}

		if node.Right != nil {
			if node.Left == nil {
				builder.WriteString("()")
			}
			preOrder(node.Right)
		}

		if node != root {
			builder.WriteString(")")
		}
	}

	preOrder(root)

	return builder.String()
}

/* 2039. 网络空闲的时刻 */
func networkBecomesIdle(edges [][]int, patience []int) int {

	graph := NewGraph(edges)

	queue := Queue{}
	queue.offer(0)

	seen := newIntSet()
	seen.add(0)
	time := 0
	for distance := 1; !queue.isEmpty(); distance++ {
		for i, size := 0, queue.len(); i < size; i++ {
			vertex := queue.poll().(int)
			seen.add(vertex)
			for _, next := range graph.next[vertex] {
				if seen.contains(next) {
					continue
				}
				queue.offer(next)
				seen.add(next)
				time = Max(time, (distance*2-1)/patience[next]*patience[next]+distance*2+1)
			}
		}
	}

	return time
}

/* 653. 两数之和 IV - 输入 BST */
func findTarget(root *TreeNode, k int) bool {

	vals := map[int]bool{}

	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {

		if node.Left != nil && dfs(node.Left) {
			return true
		}

		if vals[k-node.Val] {
			return true
		}

		vals[node.Val] = true

		if node.Right != nil && dfs(node.Right) {
			return true
		}

		return false
	}

	return dfs(root)
}

/* 2038. 如果相邻两个颜色均相同则删除当前颜色 */
func winnerOfGame(colors string) bool {

	// count表示爱丽丝比鲍勃可多操作的次数
	count := 0

	cur, colorCount := 'A', 0
	for _, color := range colors {
		if cur == color {
			colorCount++
			if colorCount >= 3 {
				if cur == 'A' {
					count++
				} else {
					count--
				}
			}
		} else {
			cur, colorCount = color, 1
		}
	}

	return count > 0
}

/* 440. 字典序的第K小数字 */
func findKthNumber(n int, k int) int {

	return 1
}

/* 661. 图片平滑器 */
func imageSmoother(img [][]int) [][]int {

	m, n := len(img), len(img[0])
	ret := make([][]int, m)
	for i := 0; i < m; i++ {
		ret[i] = make([]int, n)
		for j := 0; j < n; j++ {
			num, sum := 0, 0
			for x := i - 1; x <= i+1; x++ {
				for y := j - 1; y <= j+1; y++ {
					if x >= 0 && x < m && y >= 0 && y < n {
						num++
						sum += img[x][y]
					}
				}
			}
			ret[i][j] = sum / num
		}
	}

	return ret
}

/* 172. 阶乘后的零 */
func trailingZeroes(n int) int {

	ans := 0
	for i := 5; i <= n; i += 5 {
		for x := i; x%5 == 0; x /= 5 {
			ans++
		}
	}

	return ans
}

/* 2024. 考试的最大困扰度 */
func maxConsecutiveAnswers(answerKey string, k int) int {

	n := len(answerKey)

	tCount := make([]int, n+1)
	fCount := make([]int, n+1)

	// 分别记录T和F的前缀和
	for i, key := range answerKey {
		if key == 'T' {
			tCount[i+1] = tCount[i] + 1
			fCount[i+1] = fCount[i]
		} else {
			tCount[i+1] = tCount[i]
			fCount[i+1] = fCount[i] + 1
		}
	}

	search := func(count []int) int {
		max := 0
		// 枚举左端点
		for i := 0; i < n; i++ {
			// 剩余可能最大值小于max, 可以提前结束循环
			if n-i <= max {
				break
			}
			l, r := i, n-1
			// 二分查找右端点
			for l < r {
				mid := l + (r-l+1)/2
				// k = 区间长度 - T或F的个数
				ctn := (mid - i + 1) - (count[mid+1] - count[i])
				if ctn > k {
					r = mid - 1
				} else {
					l = mid
				}
			}
			max = Max(max, r-i+1)
		}
		return max
	}

	ans := Max(search(tCount), search(fCount))

	return ans
}

/* 796. 旋转字符串 */
func rotateString(s string, goal string) bool {

	n := len(s)
	if len(goal) != n {
		return false
	}

	for start := 0; start < n; start++ {
		equal := true
		for i := 0; i < n; i++ {
			j := (i + start + n) % n
			if goal[i] != s[j] {
				equal = false
				break
			}
		}
		if equal {
			return true
		}
	}

	return false
}

/* 429. N 叉树的层序遍历 */
func levelOrder(root *Node) [][]int {

	if root == nil {
		return [][]int{}
	}

	var ans [][]int

	queue := []*Node{root}

	for len(queue) > 0 {

		var vals []int
		for i, size := 0, len(queue); i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			vals = append(vals, node.Val)
			queue = append(queue, node.Children...)
		}

		ans = append(ans, vals)
	}

	return ans
}

/* 357. 统计各位数字都不同的数字个数 */
func countNumbersWithUniqueDigits(n int) int {

	if n == 0 {
		return 1
	}

	// f[i]表示长度为i的各位数字都不同的数字个数
	f := make([]int, n)
	f[0] = 10

	ans := f[0]
	for i := 1; i < n; i++ {
		f[i] = 9
		for j := 9; j >= 9-(i+1)+2; j-- {
			f[i] *= j
		}
		ans += f[i]
	}

	return ans
}

/* 427. 建立四叉树 */
func construct(grid [][]int) *QuadNode {

	n := len(grid)

	preSum := make([][]int, n+1)
	preSum[0] = make([]int, n+1)

	for i, row := range grid {
		preSum[i+1] = make([]int, n+1)
		for j, v := range row {
			preSum[i+1][j+1] = preSum[i][j+1] + preSum[i+1][j] - preSum[i][j] + v
		}
	}

	var divide func(int, int, int, int) *QuadNode
	divide = func(x1, y1, x2, y2 int) *QuadNode {
		sum := preSum[y2][x2] - preSum[y1][x2] - preSum[y2][x1] + preSum[y1][x1]
		// 矩阵的值全相同
		if sum == 0 {
			return &QuadNode{Val: false, IsLeaf: true}
		}
		w, h := x2-x1, y2-y1
		if sum == w*h {
			return &QuadNode{Val: true, IsLeaf: true}
		}
		midX, midY := (x1+x2)/2, (y1+y2)/2
		return &QuadNode{
			true, false,
			divide(x1, y1, midX, midY),
			divide(midX, y1, x2, midY),
			divide(x1, midY, midX, y2),
			divide(midX, midY, x2, y2),
		}
	}

	return divide(0, 0, n, n)
}

/* 713. 乘积小于 K 的子数组 */
func numSubarrayProductLessThanK(nums []int, k int) int {

	n := len(nums)
	product := 1
	ans := 0
	for i, j := 0, 0; i < n; i++ {
		product *= nums[i]
		for ; j <= i && product >= k; j++ {
			product /= nums[j]
		}
		ans += i - j + 1
	}

	return ans
}

/* 1403. 非递增顺序的最小子序列 */
func minSubsequence(nums []int) []int {

	n := len(nums)

	sort.Ints(nums)

	sum := 0
	for _, num := range nums {
		sum += num
	}

	var ans []int
	for i, s := n-1, 0; i >= 0; i-- {
		ans = append(ans, nums[i])
		s += nums[i]
		if s > sum-s {
			break
		}
	}

	return ans
}
