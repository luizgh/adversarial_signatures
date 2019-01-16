import matlab.engine
import numpy as np
from scipy.io import savemat
import uuid
import os


class CLBP:
    def __init__(self, tmp_folder: str = '/dev/shm'):
        """ CLBP feature extractor, using the author's matlab code:

        http://www.comp.polyu.edu.hk/~cslzhang/code/CLBP.rar

        Guo, Zhenhua, Lei Zhang, and David Zhang. "A completed modeling of local binary
        pattern operator for texture classification." IEEE Transactions on Image
        Processing 19.6 (2010): 1657-1663.

        Parameters:
        -----------
        tmp_folder: str
            Temporary folder used in the python->matlab communication
        """

        # Add current folder to MATLABPATH (so that Matlab finds the .m scripts)
        current_folder = os.path.dirname(os.path.realpath(__file__))
        os.environ['MATLABPATH'] = current_folder

        # Initialize the engine
        self.eng = matlab.engine.start_matlab()

        # We will use a temporary file to send the image to matlab
        # (faster than passing the image as argument to the function call)
        unique_filename = str(uuid.uuid4()) + '.mat'
        self.tmp_file = os.path.join(tmp_folder, unique_filename)

    def __call__(self, img: np.ndarray):
        """ Returns the CLBP_SMCH code (P=8, R=1) for an image

        Parameters
        ----------
        img: np.ndarray
            The input image

        Returns
        -------
        np.ndarray
            The CLBP features for the image

        """
        savemat(self.tmp_file, {'Gray': img})
        features = self.eng.extract_clbp_smch(self.tmp_file)
        return np.asarray(features)

    def close(self):
        """ Removes the temporary file used for the python->matlab communication"""
        try:
            os.unlink(self.tmp_file)
        except:
            pass
