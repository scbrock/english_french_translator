#import tensorflow as tf
import numpy as np

#print('success')
#print('tf version',tf.__version__)

a = np.array([
    [[1],[2],[3],[4],[4]],
    [[1],[2],[3],[4],[4]],
    [[1],[2],[3],[4],[4]]
])

b = np.array([
    [[1],[2],[3],[4],[3]],
    [[1],[2],[3],[4],[4]],
    [[1],[2],[3],[4],[4]]
])


comparison_matrix = np.equal(a,b)
comp_sent = [np.alltrue(s) for s in np.equal(a,b)]
print(np.equal(a,b))
print(np.sum(np.equal(a,b)))
print(np.product(np.shape(np.equal(a,b))))

print([np.alltrue(s) for s in np.equal(a,b)])

print('sentence accuracy on test set:', np.sum(comp_sent)/len(comp_sent))
print('word accuracy on test set:', np.sum(comparison_matrix)/np.product(comparison_matrix.shape))
