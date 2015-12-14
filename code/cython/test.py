from dana import *
from guthrie_modelc import *

src = np.ones((256,256))
tar = np.zeros((16,16))
w = AscToAscWeights(src.shape, tar.shape)

c = DenseConnection(src, tar, w)

print np.einsum('...j,j',w, src.ravel())
