package function

import (
	"math"
	"sort"
	"strconv"
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

func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 求两个数的最大公约数 */
func gcd(a, b int) int {

	for b != 0 {
		a, b = b, a%b
	}

	return a
}
