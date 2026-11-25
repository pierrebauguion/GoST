package lu

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

// GOST_DUMPBASIS=path writes the 50th factorized basis (debug/benchmark).
var dumpPath = os.Getenv("GOST_DUMPBASIS")
var dumpCount int

func dumpBasis(m int, cols ColumnFunc) {
	dumpCount++
	if dumpCount != 50 {
		return
	}
	fh, err := os.Create(dumpPath)
	if err != nil {
		return
	}
	defer fh.Close()
	w := bufio.NewWriter(fh)
	defer w.Flush()
	fmt.Fprintln(w, m)
	for k := 0; k < m; k++ {
		idx, val := cols(k)
		fmt.Fprint(w, len(idx))
		for p, i := range idx {
			fmt.Fprint(w, " ", i, " ", strconv.FormatFloat(val[p], 'g', -1, 64))
		}
		fmt.Fprintln(w)
	}
}
