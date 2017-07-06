import numpy as np
from numpy.random_intel import RandomState
from perf import measure_and_print

rnd = RandomState(7777, brng='MT19937')
arr3 = rnd.randn(129, 512, 521)
arr2 = rnd.randn(3025, 4284)
arr1 = rnd.randn(2*10**6)

arr3f = np.array(rnd.randn(129, 512, 521), dtype=np.float32)
arr2f = np.array(rnd.randn(3025, 4284), dtype=np.float32)
arr1f = np.array(rnd.randn(2*10**6), dtype=np.float32)


measure_and_print(np.fft.fft, arr1, dict())
measure_and_print(np.fft.fft, arr1f, dict())

###

measure_and_print(np.fft.fft, arr2, dict(axis=0))
measure_and_print(np.fft.fft, arr2f, dict(axis=0))

measure_and_print(np.fft.fft, arr2, dict(axis=1))
measure_and_print(np.fft.fft, arr2f, dict(axis=1))

measure_and_print(np.fft.fft2, arr2, dict() )
measure_and_print(np.fft.fft2, arr2f, dict() )

measure_and_print(np.fft.fft2, arr3, dict() )
measure_and_print(np.fft.fft2, arr3f, dict() )

###

measure_and_print(np.fft.fftn, arr3, dict() )
measure_and_print(np.fft.fftn, arr3f, dict() )
