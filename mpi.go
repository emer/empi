// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copied and significantly modified from: https://github.com/cpmech/gosl/mpi
// Copyright 2016 The Gosl Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package empi

/*
#cgo pkg-config: ompi
#include "mpi.h"

MPI_Comm     World     = MPI_COMM_WORLD;
MPI_Datatype INT64     = MPI_LONG;
MPI_Datatype FLOAT64   = MPI_DOUBLE;
MPI_Datatype FLOAT32   = MPI_FLOAT;
MPI_Datatype COMPLEX128 = MPI_DOUBLE_COMPLEX;
MPI_Status*  StIgnore   = MPI_STATUS_IGNORE;
*/
import "C"

import (
	"fmt"
	"log"
	"unsafe"
)

// set LogErrors to control whether MPI errors are automatically logged or not
var LogErrors = true

// Error takes an MPI error code and returns an appropriate error
// value -- either nil if no error, or the MPI error message
// with given context
func Error(ec C.int, ctxt string) error {
	if ec == C.MPI_SUCCESS {
		return nil
	}
	var rsz C.int
	str := C.malloc(C.size_t(C.MPI_MAX_ERROR_STRING))

	C.MPI_Error_string(C.int(ec), (*C.char)(str), &rsz)
	gstr := C.GoStringN((*C.char)(str), rsz)
	// C.free(str)
	err := fmt.Errorf("MPI Error: %d %s %s", ec, ctxt, gstr)
	if LogErrors {
		log.Println(err)
	}
	return err
}

// Op is an aggregation operation: Sum, Min, Max, etc
type Op int

const (
	OpSum  Op = C.int(C.MPI_SUM)
	OpMax  Op = C.MPI_MAX
	OpMin  Op = C.MPI_MIN
	OpProd Op = C.MPI_PROD
)

const (
	// Root is the rank 0 node -- it is more semantic to use this
	Root int = 0
)

// IsOn tells whether MPI is on or not
//  NOTE: this returns true even after Stop
func IsOn() bool {
	var flag C.int
	C.MPI_Initialized(&flag)
	if flag != 0 {
		return true
	}
	return false
}

// Init initialises MPI
func Init() {
	C.MPI_Init(nil, nil)
}

// InitThreadSafe initialises MPI thread safe
func InitThreadSafe() error {
	var r int32
	C.MPI_Init_thread(nil, nil, C.MPI_THREAD_MULTIPLE, (*C.int)(unsafe.Pointer(&r)))
	if r != C.MPI_THREAD_MULTIPLE {
		return fmt.Errorf("MPI_THREAD_MULTIPLE can't be set: got %d", r)
	}
	return nil
}

// Finalize finalises MPI (frees resources, shuts it down)
func Finalize() {
	C.MPI_Finalize()
}

// WorldRank returns this proc's rank/ID within the World communicator
func WorldRank() (rank int) {
	var r int32
	C.MPI_Comm_rank(C.World, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// WorldSize returns the number of procs in the World communicator
func WorldSize() (size int) {
	var s int32
	C.MPI_Comm_size(C.World, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

// Comm is the MPI communicator -- all MPI communication operates as methods
// on this struct.  It holds the MPI_Comm communicator and MPI_Group for
// sub-World group communication.
type Comm struct {
	comm  C.MPI_Comm
	group C.MPI_Group
}

// NewComm creates a new communicator.
// if ranks is nil, communicator is for World (all active procs).
// otherwise, defined a group-level commuicator for given ranks.
func NewComm(ranks []int) (*Comm, error) {
	cm := &Comm{}
	if len(ranks) == 0 {
		cm.comm = C.World
		return cm, Error(C.MPI_Comm_group(C.World, &cm.group), "MPI_Comm_group")
	}
	rs := make([]int32, len(ranks))
	for i := 0; i < len(ranks); i++ {
		rs[i] = int32(ranks[i])
	}
	n := C.int(len(ranks))
	r := (*C.int)(unsafe.Pointer(&rs[0]))
	var wgroup C.MPI_Group
	C.MPI_Comm_group(C.World, &wgroup)
	C.MPI_Group_incl(wgroup, n, r, &cm.group)
	return cm, Error(C.MPI_Comm_create(C.World, cm.group, &cm.comm), "Comm_create")
}

// Rank returns the rank/ID for this proc
func (cm *Comm) Rank() (rank int) {
	var r int32
	C.MPI_Comm_rank(cm.comm, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// Size returns the number of procs in this communicator
func (cm *Comm) Size() (size int) {
	var s int32
	C.MPI_Comm_size(cm.comm, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

// Abort aborts MPI
func (cm *Comm) Abort() error {
	return Error(C.MPI_Abort(cm.comm, 0), "Abort")
}

// Barrier forces synchronisation
func (cm *Comm) Barrier() error {
	return Error(C.MPI_Barrier(cm.comm), "Barrier")
}

//////////////////////////////////////////////////////
//   Send / Recv

// Send64 sends values to proc
func (cm *Comm) Send64(toProc int, vals []float64) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Send(buf, C.int(len(vals)), C.FLOAT64, C.int(toProc), 10000, cm.comm), "Send64")
}

// Recv64 receives values from proc fmProc
func (cm *Comm) Recv64(vals []float64, fmProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Recv(buf, C.int(len(vals)), C.FLOAT64, C.int(fmProc), 10000, cm.comm, C.StIgnore), "Recv64")
}

// Send32 sends values to proc
func (cm *Comm) Send32(toProc int, vals []float32) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Send(buf, C.int(len(vals)), C.FLOAT32, C.int(toProc), 10000, cm.comm), "Send32")
}

// Recv32 receives values from proc fmProc
func (cm *Comm) Recv32(vals []float32, fmProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Recv(buf, C.int(len(vals)), C.FLOAT32, C.int(fmProc), 10000, cm.comm, C.StIgnore), "Recv32")
}

// SendC128 sends values to proc toProc
func (cm *Comm) SendC128(vals []complex128, toProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Send(buf, C.int(len(vals)), C.COMPLEX128, C.int(toProc), 10001, cm.comm), "SendC128")
}

// RecvC128 receives values from proc fmProc
func (cm *Comm) RecvC128(vals []complex128, fmProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Recv(buf, C.int(len(vals)), C.COMPLEX128, C.int(fmProc), 10001, cm.comm, C.StIgnore), "RecvC128")
}

// SendI64 sends values to proc toProc
func (cm *Comm) SendI64(vals []int, toProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Send(buf, C.int(len(vals)), C.INT64, C.int(toProc), 10002, cm.comm), "SendI64")
}

// RecvI64 receives values from proc fmProc
func (cm *Comm) RecvI64(vals []int, fmProc int) error {
	buf := unsafe.Pointer(&vals[0])
	return Error(C.MPI_Recv(buf, C.int(len(vals)), C.INT64, C.int(fmProc), 10002, cm.comm, C.StIgnore), "RecvI64")
}

//////////////////////////////////////////////////////
//   Bcast

// BcastFrom64 broadcasts float64 slice from given proc to all other procs
func (cm *Comm) BcastFrom64(from int, x []float64) error {
	buf := unsafe.Pointer(&x[0])
	return Error(C.MPI_Bcast(buf, C.int(len(x)), C.FLOAT64, from, cm.comm), "Bcast64")
}

// BcastFrom32 broadcasts float32 slice from given proc to all other procs
func (cm *Comm) BcastFrom32(from int, x []float32) error {
	buf := unsafe.Pointer(&x[0])
	return Error(C.MPI_Bcast(buf, C.int(len(x)), C.FLOAT32, C.int(from), cm.comm), "Bcast32")
}

// BcastFromC128 broadcasts complex128 slice from given proc to all other procs
func (cm *Comm) BcastFromC128(from int, x []complex128) error {
	buf := unsafe.Pointer(&x[0])
	return Error(C.MPI_Bcast(buf, C.int(len(x)), C.COMPLEX128, C.int(from), cm.comm), "BcastC128")
}

//////////////////////////////////////////////////////
//   Reduce

// Reduce64 reduces all values in 'orig' to 'dest' in given proc
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) Reduce64(toProc int, op Op, dest, orig []float64) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), C.FLOAT64, C.MPI_Op(C.int(op)), C.int(toProc), cm.comm), "Reduce64")
}

// Reduce32 reduces all values in 'orig' to 'dest' in given proc
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) Reduce32(toProc int, op Op, dest, orig []float32) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), C.FLOAT32, C.MPI_Op(op), C.int(toProc), cm.comm), "Reduce32")
}

// ReduceC128 reduces all values in 'orig' to 'dest' in given proc
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) ReduceC128(toProc int, op Op, dest, orig []complex128) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), C.COMPLEX128, C.MPI_Op(op), C.int(toProc), cm.comm), "ReduceC128")
}

// ReduceI64 reduces all values in 'orig' to 'dest' in given proc
// using given operation.  I64 assumes 64bit int value
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) ReduceI64(toProc int, op Op, dest, orig []int) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), C.INT64, C.MPI_Op(op), C.int(toProc), cm.comm), "ReduceI64")
}

//////////////////////////////////////////////////////
//   Allreduce

// AllReduce64 combines all values on all procs from orig into dest
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) AllReduce64(op Op, dest, orig []float64) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), C.FLOAT64, C.MPI_Op(op), cm.comm), "AllReduce64")
}

// AllReduce32 combines all values on all procs from orig into dest
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) AllReduce32(op Op, dest, orig []float32) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), C.FLOAT32, C.MPI_Op(op), cm.comm), "AllReduce32")
}

// AllReduceC128 combines all values on all procs from orig into dest
// using given operation.
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) AllReduceC128(op Op, dest, orig []complex128) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), C.COMPLEX128, C.MPI_Op(op), cm.comm), "AllReduceC128")
}

// AllReduceI64 combines all values on all procs from orig into dest
// using given operation.  I64 assumes 64bit int value
// IMPORTANT: orig and dest must be different slices
func (cm *Comm) AllReduceI64(op Op, dest, orig []int) error {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	return Error(C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), C.INT64, C.MPI_Op(op), cm.comm), "AllReduceI64")
}
