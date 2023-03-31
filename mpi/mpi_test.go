package mpi

import (
	"testing"
)

func TestMPI(t *testing.T) {
	Init()
	defer Finalize()
	comm, err := NewComm(nil)
	if err != nil {
		t.Fatal(err)
	}
	comm.Rank()
	if err = comm.SendF64(0, 0, []float64{0}); err != nil {
		t.Fatal(err)
	}
}
