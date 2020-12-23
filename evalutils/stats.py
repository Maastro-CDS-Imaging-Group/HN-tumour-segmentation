import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm


EPSILON = 0.0001


class SPP():
    """
    Implementation of the Statistical Performance Profile method for performance estimation (Yang et al., 2012)
    """
    def __init__(self, patient_ids, output_dir):                
        self.patient_ids = patient_ids
        self.output_dir = output_dir

        self.alpha = None
        self.beta = None   
        self.n_iterations = 100

    
    def estimate_performance(self, patient_volume_overlap_dict):
        '''
        Estimate alpha and beta
        '''
        perf_list = []
        intersections_list = []
        unions_list = []
        for p_id in self.patient_ids:
            perf_list.append(patient_volume_overlap_dict[p_id][1])
            intersections_list.append(patient_volume_overlap_dict[p_id][2])
            unions_list.append(patient_volume_overlap_dict[p_id][3])
            
        # Estimation loop
        alpha = 1  # Initial values
        beta = 1   #
        for k in range(self.n_iterations):

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
        
        perf_mean = alpha / (alpha + beta)
        perf_stddev = math.sqrt(alpha*beta / ((alpha+beta)**2 * (alpha+beta+1)))

        perf_distribution_info = {'alpha': float(self.alpha),
                                  'beta': float(self.beta),
                                  'performance-mean': float(perf_mean),
                                  'performance-stddev': float(perf_stddev)}
        return perf_distribution_info


    def plot_performance(self):
        '''
        Plot the PDF
        '''
        xs = np.linspace(0, 1, 1000)
        fs = beta_distribution_pdf(xs, self.alpha, self.beta)
        plt.plot(xs, fs, 'b-')
        plt.xlabel("Volume Overlap")
        plt.ylabel("Probability Density Function")

        perf_mean = self.alpha / (self.alpha + self.beta)
        plt.plot([perf_mean, perf_mean], [0, beta_distribution_pdf(perf_mean, self.alpha, self.beta)], 'k--')

        plt.title("Statistical Performance Profile")
        plt.savefig(f"{self.output_dir}/SPP_plot.png")



def compute_norm_constant(alpha, beta):
    '''
    Calculate 1 / B(alpha, beta)
    '''
    def B_integrand(t, alpha, beta):
        return t**(alpha-1) * (1-t)**(beta-1)

    B = integrate.quad(B_integrand, 0, 1, args=(alpha, beta))[0]
    norm_constant = 1 / B
    return norm_constant


def beta_distribution_pdf(x, alpha, beta):
    norm_constant = compute_norm_constant(alpha, beta)
    f = norm_constant * x**(alpha-1) * (1-x)**(beta-1)
    return f




if __name__ == '__main__':
    data_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crS_rs113_hecktor_nii"
    predictions_dir = "/home/zk315372/Chinmay/hn_experiment_results/hecktor-crS_rs113/predicted"
    patient_id_filepath = "./hecktor_meta/patient_IDs_train.txt"
    subset_name = "CHUM"

    spp = SPP(data_dir, predictions_dir, patient_id_filepath, subset_name)
    spp.estimate_performance()
    spp.plot_performance()