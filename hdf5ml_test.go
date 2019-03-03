package hdf5ml

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type pixel [3]uint8
type img64 [64][64]pixel
type label int64

const (
	fName   = "./train_catvnoncat.h5"
	xDSName = "train_set_x"
	yDSName = "train_set_y"
	cDSName = "list_classes"
)

func TestLoadSet(t *testing.T) {
	m := NewMLHDF5(fName, cDSName, xDSName, yDSName)
	d := m.GetDSName(XTYPE)
	assert.Equal(t, xDSName, d)
	d = m.GetDSName(YTYPE)
	assert.Equal(t, yDSName, d)
	d = m.GetDSName(CLASSESTYPE)
	assert.Equal(t, cDSName, d)

	dims, maxDims, nDims, nPoints, typeP, size, err := m.GetDSetValues(YTYPE)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v %v %v %v %v %v %v", dims, maxDims, nDims, nPoints, typeP, size, err)
	trainy := make([]label, dims[0])
	err = m.GetHDF5Data(YTYPE, &trainy)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", trainy[0])

	dims, maxDims, nDims, nPoints, typeP, size, err = m.GetDSetValues(XTYPE)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v %v %v %v %v %v %v", dims, maxDims, nDims, nPoints, typeP, size, err)
	trainx := make([]img64, dims[0])
	err = m.GetHDF5Data(XTYPE, &trainx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", trainx[0][0][0])

	dims, maxDims, nDims, nPoints, typeP, size, err = m.GetDSetValues(CLASSESTYPE)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v %v %v %v %v %v %v", dims, maxDims, nDims, nPoints, typeP, size, err)
	classes := make([]byte, dims[0]*size)
	err = m.GetHDF5Data(CLASSESTYPE, &classes)
	if err != nil {
		t.Fatal(err)
	}
	str := make([]string, dims[0])
	var i uint
	for i = 0; i < dims[0]; i++ {
		str[i] = string(classes[i*size : i*size+size])
	}
	t.Logf("%v", str)
}
