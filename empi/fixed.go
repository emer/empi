// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package empi

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// mpi.FixedTable is an MPI-enabled version of the env.FixedTable, which is
// a basic Env that manages patterns from an etable.Table, with
// either sequential or permuted random ordering, and uses standard Trial / Epoch
// TimeScale counters to record progress and iterations through the table.
// It also records the outer loop of Run as provided by the model.
// It uses an IdxView indexed view of the Table, so a single shared table
// can be used across different environments, with each having its own unique view.
// The MPI version distributes trials across MPI procs, in the Order list.
// It is ESSENTIAL that the number of trials (rows) in Table is
// evenly divisible by number of MPI procs!
// If all nodes start with the same seed, it should remain synchronized.
type FixedTable struct {
	Nm         string           `desc:"name of this environment"`
	Dsc        string           `desc:"description of this environment"`
	Table      *etable.IdxView  `desc:"this is an indexed view of the table with the set of patterns to output -- the indexes are used for the *sequential* view so you can easily sort / split / filter the patterns to be presented using this view -- we then add the random permuted Order on top of those if !sequential"`
	Sequential bool             `desc:"present items from the table in sequential order (i.e., according to the indexed view on the Table)?  otherwise permuted random order"`
	Order      []int            `desc:"permuted order of items to present if not sequential -- updated every time through the list"`
	Run        env.Ctr          `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr          `view:"inline" desc:"number of times through entire set of patterns"`
	Trial      env.Ctr          `view:"inline" desc:"current ordinal item in Table -- if Sequential then = row number in table, otherwise is index in Order list that then gives row number in Table"`
	TrialName  env.CurPrvString `desc:"if Table has a Name column, this is the contents of that"`
	GroupName  env.CurPrvString `desc:"if Table has a Group column, this is contents of that"`
	NameCol    string           `desc:"name of the Name column -- defaults to 'Name'"`
	GroupCol   string           `desc:"name of the Group column -- defaults to 'Group'"`
	TrialSt    int              `desc:"for MPI, trial we start each epoch on, as index into Order"`
	TrialEd    int              `desc:"for MPI, trial number we end each epoch before (i.e., when ctr gets to Ed, restarts)"`
}

func (ft *FixedTable) Name() string { return ft.Nm }
func (ft *FixedTable) Desc() string { return ft.Dsc }

func (ft *FixedTable) Validate() error {
	if ft.Table == nil || ft.Table.Table == nil {
		return fmt.Errorf("env.FixedTable: %v has no Table set", ft.Nm)
	}
	if ft.Table.Table.NumCols() == 0 {
		return fmt.Errorf("env.FixedTable: %v Table has no columns -- Outputs will be invalid", ft.Nm)
	}
	return nil
}

func (ft *FixedTable) Init(run int) {
	if ft.NameCol == "" {
		ft.NameCol = "Name"
	}
	if ft.GroupCol == "" {
		ft.GroupCol = "Group"
	}
	ft.Run.Scale = env.Run
	ft.Epoch.Scale = env.Epoch
	ft.Trial.Scale = env.Trial
	ft.Run.Init()
	ft.Epoch.Init()
	ft.Trial.Init()
	ft.Run.Cur = run
	ft.NewOrder()
	ft.Trial.Cur = ft.TrialSt - 1 // init state -- key so that first Step() = ft.TrialSt
}

// NewOrder sets a new random Order based on number of rows in the table.
func (ft *FixedTable) NewOrder() {
	np := ft.Table.Len()
	ft.Order = rand.Perm(np) // always start with new one so random order is identical
	// and always maintain Order so random number usage is same regardless, and if
	// user switches between Sequential and random at any point, it all works..
	nproc := mpi.WorldSize()
	pt := np / nproc
	if np%nproc != 0 {
		log.Printf("mpi.FixedTable: number of table rows: %d is not an even multiple of number of MPI procs: %d -- must be!\n", np, nproc)
	}
	ft.TrialSt = pt * mpi.WorldRank()
	ft.TrialEd = ft.TrialSt + pt
	ft.Trial.Max = ft.TrialEd
}

// PermuteOrder permutes the existing order table to get a new random sequence of inputs
// just calls: erand.PermuteInts(ft.Order)
func (ft *FixedTable) PermuteOrder() {
	erand.PermuteInts(ft.Order)
}

// Row returns the current row number in table, based on Sequential / perumuted Order and
// already de-referenced through the IdxView's indexes to get the actual row in the table.
func (ft *FixedTable) Row() int {
	if ft.Sequential {
		return ft.Table.Idxs[ft.Trial.Cur]
	}
	return ft.Table.Idxs[ft.Order[ft.Trial.Cur]]
}

func (ft *FixedTable) SetTrialName() {
	if nms := ft.Table.Table.ColByName(ft.NameCol); nms != nil {
		rw := ft.Row()
		if rw >= 0 && rw < nms.Len() {
			ft.TrialName.Set(nms.StringVal1D(rw))
		}
	}
}

func (ft *FixedTable) SetGroupName() {
	if nms := ft.Table.Table.ColByName(ft.GroupCol); nms != nil {
		rw := ft.Row()
		if rw >= 0 && rw < nms.Len() {
			ft.GroupName.Set(nms.StringVal1D(rw))
		}
	}
}

func (ft *FixedTable) Step() bool {
	ft.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	if ft.Trial.Incr() { // if true, hit max, reset to 0
		ft.PermuteOrder()
		ft.Epoch.Incr()
	}
	ft.SetTrialName()
	ft.SetGroupName()
	return true
}

func (ft *FixedTable) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ft *FixedTable) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ft.Run.Query()
	case env.Epoch:
		return ft.Epoch.Query()
	case env.Trial:
		return ft.Trial.Query()
	}
	return -1, -1, false
}

func (ft *FixedTable) States() env.Elements {
	els := env.Elements{}
	els.FromSchema(ft.Table.Table.Schema())
	return els
}

func (ft *FixedTable) State(element string) etensor.Tensor {
	et, err := ft.Table.Table.CellTensorTry(element, ft.Row())
	if err != nil {
		log.Println(err)
	}
	return et
}

func (ft *FixedTable) Actions() env.Elements {
	return nil
}

func (ft *FixedTable) Action(element string, input etensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*FixedTable)(nil)
