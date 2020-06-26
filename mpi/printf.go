// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpi

import "fmt"

// Printf does fmt.Printf only on the 0 rank node
func Printf(fs string, pars ...interface{}) {
	if WorldRank() != 0 {
		return
	}
	fmt.Printf(fs, pars...)
}
