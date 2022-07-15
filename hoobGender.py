import os,sys
import logging
import pickle
import argparse
from dataclasses import dataclass

from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pysam


@dataclass
class Ygenes:
    '''
    The male-specific genes of the human Y chromosome (MSY) - hg38
    '''
    HSFY1:  str = "chrY:18544690-18590963"
    BPY2:   str	= "chrY:22982262-23007465"
    BPY2B:  str	= "chrY:24616003-24641207"
    BPY2C:  str	= "chrY:25028900-25054104"
    XKRY:   str	= "chrY:17766979-17772560"
    PRY:    str	= "chrY:22488290-22518303"
    PRY2:   str	= "chrY:22069755-22098007"
        

class Helpers:

    def __init__(self, out, log) -> None:
        self.out = out
        self.log = log
        self.genLogger()


    def genLogger(self) -> None:
        levels = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warn': logging.WARNING,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG
            }
        level = levels.get(self.log.lower())
        logging.basicConfig(level=level,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename= os.path.join(self.out, 'hoobGender.log'),
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        logging.info('Logger initialized')


    def exception_handler(func) -> None:
        def inner_function(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logging.info("\t\tError in: {}, Illegal path or object\n".format(func.__name__))
        return inner_function


    @exception_handler
    def spitPickle(self, obj, path) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


    @exception_handler
    def _exists(self, path) -> bool:
        return os.path.exists(path)


    @exception_handler
    def loadPickle(self, path) -> object:
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class HoobGender:
    
    def __init__(self, args) -> None:
        self.bam_path = args.bam
        self.sample_name = args.sample
        self.fractions_path = args.fractions
        self.out = args.out
        self.log = args.log

        try:
            os.makedirs(self.out, exist_ok=True)
        except Exception as e:
            logging.info("\t\tError in: HoobGender __init__, unable to create output directory\n")
            sys.exit(1)

        self.helpers = Helpers(self.out, self.log)
        logging.info('HoobGender started {}'.format(self.sample_name))

        self.Y = Ygenes()

        if self.helpers._exists(self.fractions_path):
            self.fractions = self.helpers.loadPickle(self.fractions_path)
            logging.info(self.fractions)
        else:
            logging.info('{} does not exist, exiting'.format(self.fractions_path))
            sys.exit(1)
        
        self.y_fraction_specific()
        self.calculate_threshold()
        
    
    def y_fraction_specific(self) -> None:
        samfile = pysam.AlignmentFile(self.bam_path, "rb")
        count_spec = 0
        for gene,coord in self.Y.__dict__.items():
            for read in samfile.fetch(region = coord):
                count_spec += 1
        logging.info('{} has {} reads on MSY'.format(self.bam_path, count_spec))

        count_Y = 0
        for read in samfile.fetch('chrY'):
            count_Y += 1
        if count_Y == 0:
            logging.warning('{} has 0 reads on chrY'.format(self.bam_path))
            count_Y = 1
        logging.info('{} has {} reads on chrY'.format(self.bam_path, count_Y))

        samfile.close()

        self.fraction_specific_to_Y = (count_spec * 100) / count_Y
        logging.info('{} fraction of MSY reads on chrY: {}'.format(self.bam_path, self.fraction_specific_to_Y))
    
    
    def calculate_threshold(self) -> None:
        """
        Compute a cut off with a Gaussian Mixture Model representing a bimodal gaussian distribution.
        """
        y_fractions = np.array(self.fractions.Fraction)
        gmm = GaussianMixture(n_components=2, covariance_type='full', reg_covar=1e-99, max_iter=10000, tol=1e-99)
        gmm.fit(X=y_fractions.reshape(-1, 1))
        gmm_x = np.linspace(min(y_fractions), max(y_fractions), 5000)
        gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

        sort_idd = np.argsort(gmm_x)
        sorted_gmm_y = gmm_y[sort_idd]

        local_min_i = argrelextrema(sorted_gmm_y, np.less)

        self.cut_off = gmm_x[local_min_i][0]

        #plot gmm fig
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(y_fractions, bins=50, density=True)
        ax.plot(gmm_x, gmm_y, 'r-', label='Gaussian mixture fit')
        
        ax.set_xlim([min(y_fractions), max(y_fractions)])
        ax.set_xlabel('Fraction of MSY reads on chrY')
        ax.set_ylabel('Probability density')
        ax.set_title('Gaussian mixture fit. Sample {}'.format(self.sample_name))
        plt.axvline(x=self.cut_off, linestyle='dashed', color = 'green', label = 'threshold = %.5f' % self.cut_off)
        ax.legend(loc='best')
        plt.savefig(os.path.join(self.out, 'gmm_fig.png'))
    

    def prediction(self) -> str:
        self.gender_prediction = "Male" if self.fraction_specific_to_Y > self.cut_off else "Female"
        logging.info("hoobGender prediction for {}: {}".format(self.sample_name, self.gender_prediction))

        self.fractions.loc[len(self.fractions.index)] = [self.sample_name, self.gender_prediction, self.fraction_specific_to_Y]
        self.helpers.spitPickle(self.fractions, os.path.join(self.out, 'fractions.pkl'))

        return self.gender_prediction


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bam', help='Path to BAM file', required=True, type=str)
    parser.add_argument('-s', '--sample', help='Sample name', required=True, type=str)
    parser.add_argument('-f', '--fractions', help='Path to fractions pickle file', required=True, type=str)
    
    parser.add_argument('-o', '--out', help='Output directory', required=False, default='./hoobGender', type=str)
    parser.add_argument('-log', '--log', default="info", help=("Provide logging level. Example --log warning', default='info'"))
    args = parser.parse_args()
    
    hoob = HoobGender(args)
    print("hoobGender prediction for {}: {}".format(args.sample, hoob.prediction()))
