package lu

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// GOST_BASES=dir go test -run TestFactorSpeed -v ./internal/lu
func TestFactorSpeed(t *testing.T) {
	dir := os.Getenv("GOST_BASES")
	if dir == "" {
		t.Skip()
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.basis"))
	for _, fn := range files {
		fh, _ := os.Open(fn)
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		sc.Scan()
		m, _ := strconv.Atoi(sc.Text())
		idx := make([][]int, m)
		val := make([][]float64, m)
		for k := 0; k < m; k++ {
			sc.Scan()
			fs := strings.Fields(sc.Text())
			for p := 1; p+1 < len(fs); p += 2 {
				i, _ := strconv.Atoi(fs[p])
				v, _ := strconv.ParseFloat(fs[p+1], 64)
				idx[k] = append(idx[k], i)
				val[k] = append(val[k], v)
			}
		}
		fh.Close()
		f := New(m)
		cf := func(k int) ([]int, []float64) { return idx[k], val[k] }
		f.Factorize(m, cf)
		n := 30 * benchMul
		t0 := time.Now()
		for it := 0; it < n; it++ {
			f.Factorize(m, cf)
		}
		l, u, _ := f.FillIn()
		t.Logf("%-12s m=%d %8.2fms L=%d U=%d", filepath.Base(fn), m, float64(time.Since(t0).Microseconds())/1000/float64(n), l, u)
	}
}

var benchMul = int(envFloat("GOST_BENCHMUL", 1))
