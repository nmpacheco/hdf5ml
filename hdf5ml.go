package hdf5ml

import (
	"fmt"

	"gonum.org/v1/hdf5"
)

type MLHDF5 interface {
	GetDSName(SetType) string
	GetDSetValues(SetType) ([]uint, []uint, int, int, hdf5.SpaceClass, uint, error)
	GetHDF5Data(SetType, interface{}) error
}

// MLHDF5 holds data read from FD hdf5 file
type mlhdf5 struct {
	FName string
	DSets map[SetType]string
}

// SetType defines type for dataset type
type SetType string

const (
	// XTYPE is the examples dataset type
	XTYPE SetType = "X"
	// YTYPE is the labels dataset type
	YTYPE SetType = "Y"
	// CLASSESTYPE is the classes dataset type
	CLASSESTYPE SetType = "Classes"
)

func NewMLHDF5(fName, cDSName, xSetDSName, ySetDSName string) MLHDF5 {
	return &mlhdf5{fName, map[SetType]string{CLASSESTYPE: cDSName, XTYPE: xSetDSName, YTYPE: ySetDSName}}
}

func (m *mlhdf5) GetDSName(s SetType) string {
	return m.DSets[s]
}

func (m *mlhdf5) GetDSetValues(s SetType) ([]uint, []uint, int, int, hdf5.SpaceClass, uint, error) {
	hdf5d, err := openHDF5File(m.FName, hdf5.F_ACC_RDONLY)
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	defer hdf5d.Close()

	dset, err := hdf5d.OpenDataset(m.DSets[s])
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}

	dspace := dset.Space()

	dims, maxDims, err := dspace.SimpleExtentDims()
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	nDims := dspace.SimpleExtentNDims()
	nPoints := dspace.SimpleExtentNPoints()
	typeP := dspace.SimpleExtentType()

	dtype, err := dset.Datatype()
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	size := dtype.Size()

	return dims, maxDims, nDims, nPoints, typeP, size, nil
}

func (m *mlhdf5) GetHDF5Data(s SetType, d interface{}) error {
	hdf5d, err := openHDF5File(m.FName, hdf5.F_ACC_RDONLY)
	if err != nil {
		return err
	}
	defer hdf5d.Close()

	dset, err := hdf5d.OpenDataset(m.DSets[s])
	if err != nil {
		return err
	}

	err = dset.Read(d)
	if err != nil {
		return err
	}
	return nil
}

func openHDF5File(fd string, access int) (hdf5.File, error) {

	if !hdf5.IsHDF5(fd) {
		return hdf5.File{}, fmt.Errorf("%s is not a HDF5 file", fd)
	}

	fdHDF5, err := hdf5.OpenFile(fd, access)
	if err != nil {
		return hdf5.File{}, err
	}

	return *fdHDF5, nil
}
