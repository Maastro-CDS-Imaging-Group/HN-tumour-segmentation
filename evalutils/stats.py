import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm


EPSILON = 0.0001


class SPP():
    """
    Implementation of the Statistical Performance Profile method (Yang et al., 2012)
    """
    def __init__(self, data_dir, predictions_dir, patient_id_filepath, subset_name):        
        self.data_dir = data_dir
        self.predictions_dir = predictions_dir

        with open(patient_id_filepath, 'r') as pf:
            self.patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']
        
        # CrS dataset correction
        if "crS_rs113" in self.data_dir:
            self.patient_ids.remove("CHUM010")
            self.patient_ids.remove("CHUS021")

        self.subset_name = subset_name
        self.patient_ids = [p_id for p_id in self.patient_ids if subset_name in p_id]

        self.norm_constant = None
        self.alpha = None
        self.beta = None   
        self.n_iterations = 100

    
    def estimate_performance(self):
        '''
        Estimate alpha and beta
        '''

        # Compute and cache the intersection, union and IoU for each patient
        intersections_list = []
        unions_list = []
        perf_list = []
        for p_id in self.patient_ids:
            gtv = sitk.ReadImage(f"{self.data_dir}/{p_id}_ct_gtvt.nii.gz")
            gtv = sitk.GetArrayFromImage(gtv)

            pred = sitk.ReadImage(f"{self.predictions_dir}/{p_id}_pred_gtvt.nrrd")            
            pred = sitk.GetArrayFromImage(pred)
            pred = pred >= 0.5
            # pred = gtv.copy() + np.random.choice([0,1], size=gtv.shape, p=[0.999,0.001]) # For testing purpose

            intersection = np.sum((gtv * pred)).astype(float)
            union = np.sum(np.maximum(gtv, pred)).astype(float)
            initial_perf = (intersection + EPSILON) / (union + EPSILON)

            # Store the values
            intersections_list.append(intersection)
            unions_list.append(union)
            perf_list.append(initial_perf)


        # Estimation loop
        alpha = 1  # Initial values
        beta = 1   #
        for k in tqdm(range(self.n_iterations)):

            # Update the estimated performance values using previous alpha and beta values
            for i in range(len(self.patient_ids)):
                intersection = intersections_list[i]
                union = unions_list[i]
                perf_list[i] = (intersection + alpha - 1 + EPSILON) / (union + alpha + beta - 2 + EPSILON)
                                
            # Update alpha and beta using current estimated performance values
            perf_array = np.array(perf_list)
            perf_moment_1 = np.mean(perf_array)
            perf_moment_2 = np.mean((perf_array-perf_moment_1)**2)

            moments_factor = perf_moment_1 * (1-perf_moment_1) / perf_moment_2 - 1
            alpha = perf_moment_1 * moments_factor
            beta = (1 - perf_moment_1) * moments_factor

        self.alpha = alpha
        self.beta = beta
        self._compute_norm_constant()
        perf_mean = alpha / (alpha + beta)
        perf_stddev = math.sqrt(alpha*beta / ((alpha+beta)**2 * (alpha+beta+1)))

        print(f"Alpha: {self.alpha:.3f}")
        print(f"Beta: {self.alpha:.3f}")
        print(f"Performance mean: {perf_mean:.3f}")
        print(f"Performance std dev: {perf_stddev:.3f}")


    def plot_performance(self):
        '''
        Plot the PDF
        '''
        xs = np.linspace(0, 1, 1000)
        fs = self._beta_distribution_pdf(xs)
        plt.plot(xs, fs)
        plt.savefig("./perf_plot.png")


    def _compute_norm_constant(self):
        '''
        Calculate 1 / B(alpha, beta)
        '''
        def B_integrand(t, alpha, beta):
            return t**(alpha-1) * (1-t)**(beta-1)

        B = integrate.quad(B_integrand, 0, 1, args=(self.alpha, self.beta))[0]
        self.norm_constant = 1 / B
        # self.norm_constant = 1


    def _beta_distribution_pdf(self, x):
        f = self.norm_constant * x**(self.alpha-1) * (1-x)**(self.beta-1)
        return f


if __name__ == '__main__':
    data_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crS_rs113_hecktor_nii"
    predictions_dir = "/home/zk315372/Chinmay/hn_experiment_results/hecktor-crS_rs113/predicted"
    patient_id_filepath = "./hecktor_meta/patient_IDs_train.txt"
    subset_name = "CHUM"

    spp = SPP(data_dir, predictions_dir, patient_id_filepath, subset_name)
    spp.estimate_performance()
    spp.plot_performance()