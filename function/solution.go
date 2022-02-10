package function

import (
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

/* 求两个数的最大公约数 */
func gcd(a, b int) int {

	for b != 0 {
		a, b = b, a%b
	}

	return a
}
