# eMPI: Message Passing Interface

[![Go Report Card](https://goreportcard.com/badge/github.com/emer/empi)](https://goreportcard.com/report/github.com/emer/empi)
[![GoDoc](https://godoc.org/github.com/emer/emergent?status.svg)](https://godoc.org/github.com/emer/empi)

eMPI contains Go wrappers around the MPI message passing interface for distributed memory computation, in the `empi/mpi` package.  This has no other dependencies and uses code generation to provide support for all Go types.

You must set the `mpi` build tag to actually have it build using the mpi library -- the default is to build a dummy version that has 1 proc of rank 0 always, and nop versions of all the methods.

```bash
$ go build -tags mpi
```

The `empi/empi` package has methods to support use of MPI in emergent simulations:

* Gathering `etable.Table` and `etensor.Tensor` data across processors.

* A version of env.FixedTable that divides rows of table across MPI processors.


## Bazel

After adding any new files or imports, please update the Bazel files automatically with:

```sh
# Updates BUILD.bazel files
bazel run //:gazelle -- empi
# Updates external repos in WORKSPACE.bazel
bazel run //:gazelle -- update-repos -from_file=go.mod
bazel build //...
```
