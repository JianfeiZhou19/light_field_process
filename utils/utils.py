import scipy
from .mat73 import loadmat

def load_mat(file):
    try:
        data = loadmat(file)['LF'][:, :, :, :, :3]
    except:
        data = scipy.io.loadmat(file)['LF'][:, :, :, :, :3]
    return data