// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package empi

import (
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/ints"
)

// GatherTensorRows does an MPI AllGather on given src tensor data, gathering into dest,
// using a row-based tensor organization (as in an etable.Table).
// dest will have np * src.Rows Rows, filled with each processor's data, in order.
// dest must have same overall shape as src at start, but rows will be enforced.
func GatherTensorRows(dest, src etensor.Tensor, comm *mpi.Comm) error {
	dt := src.DataType()
	if dt == etensor.STRING {
		return GatherTensorRowsString(dest.(*etensor.String), src.(*etensor.String), comm)
	}
	sr, _ := src.RowCellSize()
	dr, _ := dest.RowCellSize()
	np := mpi.WorldSize()
	dl := np * sr
	if dr != dl {
		dest.SetNumRows(dl)
		dr = dl
	}

	var err error
	switch dt {
	case etensor.BOOL:
		// todo
	case etensor.UINT8:
		dt := dest.(*etensor.Uint8)
		st := src.(*etensor.Uint8)
		err = comm.AllGatherU8(dt.Values, st.Values)
	case etensor.INT8:
		dt := dest.(*etensor.Int8)
		st := src.(*etensor.Int8)
		err = comm.AllGatherI8(dt.Values, st.Values)
	case etensor.UINT16:
		dt := dest.(*etensor.Uint16)
		st := src.(*etensor.Uint16)
		err = comm.AllGatherU16(dt.Values, st.Values)
	case etensor.INT16:
		dt := dest.(*etensor.Int16)
		st := src.(*etensor.Int16)
		err = comm.AllGatherI16(dt.Values, st.Values)
	case etensor.UINT32:
		dt := dest.(*etensor.Uint32)
		st := src.(*etensor.Uint32)
		err = comm.AllGatherU32(dt.Values, st.Values)
	case etensor.INT32:
		dt := dest.(*etensor.Int32)
		st := src.(*etensor.Int32)
		err = comm.AllGatherI32(dt.Values, st.Values)
	case etensor.UINT64:
		dt := dest.(*etensor.Uint64)
		st := src.(*etensor.Uint64)
		err = comm.AllGatherU64(dt.Values, st.Values)
	case etensor.INT64:
		dt := dest.(*etensor.Int64)
		st := src.(*etensor.Int64)
		err = comm.AllGatherI64(dt.Values, st.Values)
	case etensor.INT:
		dt := dest.(*etensor.Int)
		st := src.(*etensor.Int)
		err = comm.AllGatherInt(dt.Values, st.Values)
	case etensor.FLOAT32:
		dt := dest.(*etensor.Float32)
		st := src.(*etensor.Float32)
		err = comm.AllGatherF32(dt.Values, st.Values)
	case etensor.FLOAT64:
		dt := dest.(*etensor.Float64)
		st := src.(*etensor.Float64)
		err = comm.AllGatherF64(dt.Values, st.Values)
	}
	return err
}

// GatherTensorRowsString does an MPI AllGather on given String src tensor data,
// gathering into dest, using a row-based tensor organization (as in an etable.Table).
// dest will have np * src.Rows Rows, filled with each processor's data, in order.
// dest must have same overall shape as src at start, but rows will be enforced.
func GatherTensorRowsString(dest, src *etensor.String, comm *mpi.Comm) error {
	sr, _ := src.RowCellSize()
	dr, _ := dest.RowCellSize()
	np := mpi.WorldSize()
	dl := np * sr
	if dr != dl {
		dest.SetNumRows(dl)
		dr = dl
	}
	ssz := len(src.Values)
	dsz := len(dest.Values)
	sln := make([]int, ssz)
	dln := make([]int, dsz)
	for i, s := range src.Values {
		sln[i] = len(s)
	}
	err := comm.AllGatherInt(dln, sln)
	if err != nil {
		return err
	}
	mxlen := 0
	for _, l := range dln {
		mxlen = ints.MaxInt(mxlen, l)
	}
	if mxlen == 0 {
		return nil // nothing to transfer
	}
	sdt := make([]byte, ssz*mxlen)
	ddt := make([]byte, dsz*mxlen)
	idx := 0
	for _, s := range src.Values {
		l := len(s)
		copy(sdt[idx:idx+l], []byte(s))
		idx += mxlen
	}
	err = comm.AllGatherU8(ddt, sdt)
	idx = 0
	for i := range dest.Values {
		l := dln[i]
		s := string(ddt[idx : idx+l])
		dest.Values[i] = s
		idx += mxlen
	}
	return err
}
