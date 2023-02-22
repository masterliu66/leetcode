package function

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func AssertEqual(t *testing.T, expected interface{}, actual interface{}) {

	a := assert.New(t)

	a.Equal(expected, actual)
}

func assertLessOrEqual(t *testing.T, e1 interface{}, e2 interface{}) {

	a := assert.New(t)

	a.LessOrEqual(e1, e2)
}
