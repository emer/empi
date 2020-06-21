// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package empi wraps the Message Passing Interface for distributed memory
data sharing across a collection of processors (procs).

It also contains some useful abstractions and error logging support in Go
on top of the basic MPI_* calls invoked through cgo.

The wrapping code was copied directly from https://github.com/cpmech/gosl/mpi
we needed to change a few things (including support for float32)
and generally have this modifiable by users in case of any linking path issues.
*/
package empi
