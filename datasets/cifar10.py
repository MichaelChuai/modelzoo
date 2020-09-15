import joblib
import numpy as np
import dlutil as dl

a = list(joblib.load('/data/testdata/cifar10/cifar10.pkl'))

dt = {'tr':[[a[0]], [a[1]]], 'te':[[a[2]], [a[3]]] }

dl.write_classified_h5_from_arrays('/data/testdata/cifar10/cifar10.h5', dt, is_appending=False)


# ds = dl.DataReader('/data/testdata/cifar10/cifar10.h5').data_source
