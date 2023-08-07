package dp

func Max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func Min3(a, b, c int) int {
	return Min(Min(a, b), c)
}

func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
