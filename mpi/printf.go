// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpi

import "fmt"

// PrintAllProcs causes mpi.Printf to print on all processors -- otherwise just 0
var PrintAllProcs = false

// Printf does fmt.Printf only on the 0 rank node (see also AllPrintf to do all)
// and PrintAllProcs var to override for debugging, and print all
func Printf(fs string, pars ...interface{}) {
	if !PrintAllProcs && WorldRank() > 0 {
		return
	}
	if WorldRank() > 0 {
		fs = fmt.Sprintf("P%d: ", WorldRank()) + fs
	}
	fmt.Printf(fs, pars...)
}

// AllPrintf does fmt.Printf on all nodes, with node rank printed first
// This is best for debugging MPI itself.
func AllPrintf(fs string, pars ...interface{}) {
	fs = fmt.Sprintf("P%d: ", WorldRank()) + fs
	fmt.Printf(fs, pars...)
}

// Println does fmt.Println only on the 0 rank node (see also AllPrintln to do all)
// and PrintAllProcs var to override for debugging, and print all
func Println(fs string) {
	if !PrintAllProcs && WorldRank() > 0 {
		return
	}
	if WorldRank() > 0 {
		fs = fmt.Sprintf("P%d: ", WorldRank()) + fs
	}
	fmt.Println(fs)
}

// AllPrintln does fmt.Println on all nodes, with node rank printed first
// This is best for debugging MPI itself.
func AllPrintln(fs string) {
	fs = fmt.Sprintf("P%d: ", WorldRank()) + fs
	fmt.Println(fs)
}
