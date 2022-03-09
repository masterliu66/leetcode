package function

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
func Gcd(a, b int) int {

	for b != 0 {
		a, b = b, a%b
	}

	return a
}
