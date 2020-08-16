// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package empi

import (
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
)

// GatherTableRows does an MPI AllGather on given src table data, gathering into dest.
// dest will have np * src.Rows Rows, filled with each processor's data, in order.
// dest must be a clone of src: if not same number of cols, will be configured from src.
func GatherTableRows(dest, src *etable.Table, comm *mpi.Comm) {
	sr := src.Rows
	np := mpi.WorldSize()
	dr := np * sr
	if len(dest.Cols) != len(src.Cols) {
		dest.SetFromSchema(src.Schema(), dr)
	} else {
		dest.SetNumRows(dr)
	}
	for ci, st := range src.Cols {
		dt := dest.Cols[ci]
		GatherTensorRows(dt, st, comm)
	}
}
