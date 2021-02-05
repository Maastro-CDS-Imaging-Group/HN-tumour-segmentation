"""
Takes the performance scorecards and per patient metrics of each segmentation approach and assembles a comparison chart.

To include:
    - Combined SPP plot
    - Combined Patient vs. Dice plot
    - Box plots of Dice and Hausdorff
    - Wilcoxon signed ranked test on Dice 
"""

import os
import math
import argparse
from collections import OrderedDict
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from evalutils.stats import beta_distribution_pdf


PERFORMANCES_ROOT_DIR = "/home/zk315372/Chinmay/model_performances"
PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

SEG_APPROACHES = ('unet3d_pet', 'unet3d_ct', 'unet3d_petct', 'unet3d_latefusion', 'msam3d_petct')
SEG_DISPLAY_NAMES = {'unet3d_pet': 'PET U-Net', 'unet3d_ct': 'CT U-Net', 
                     'unet3d_petct': 'Input-level fusion U-Net', 
                     'unet3d_latefusion': 'Decision-level fusion U-Nets',
                     'msam3d_petct': 'MSAM'}


def main():
    with open(PATIENT_ID_FILEPATH, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

    # Correction for crS
    print("Warning -- Correction for crop-S data: Not considering patients CHUM010 and CHUS021 in evaluation\n")
    patient_ids.remove("CHUM010")
    patient_ids.remove("CHUS021")

    # -------------------------------------------------------------------
    # Fetch the scorecard data and patient-wise metrics for all seg approaches
    seg_approaches = SEG_APPROACHES
    scorecards_dict = {} # Model wise scorecards
    patient_metrics_dict = {} # Model wise per-patient metrics
    
    for seg_appr in seg_approaches:
        performance_dir = f"{PERFORMANCES_ROOT_DIR}/hecktor-crS_rs113/{seg_appr}"

        with open(f"{performance_dir}/scorecard.yaml", 'r') as sc_stream:
            scorecard = yaml.safe_load(sc_stream)
        scorecards_dict[seg_appr] = scorecard

        df = pd.read_csv(f"{performance_dir}/per_patient_metrics.csv")
        patient_metrics_dict[seg_appr] = df.to_dict(orient='list')

    # -------------------------------------------------------------------
    # Combine the information across seg approaches
    output_dir = f"{PERFORMANCES_ROOT_DIR}/hecktor-crS_rs113"
    # output_dir = "./temp_dir" ##

    # 1. Combined SPP plot
    plot_combined_SPP(scorecards_dict, savefig=True, output_dir=output_dir)

    # 2. Combined Patient vs. Dice plot
    plot_combined_patient_dice(patient_ids, patient_metrics_dict, savefig=True, output_dir=output_dir)

    # 3. Box plots
    boxplot_patient_metrics(patient_ids, patient_metrics_dict, metric_name='dice', savefig=True, output_dir=output_dir)


def plot_combined_SPP(scorecards_dict, savefig, output_dir):
    xs = np.linspace(0, 1, 1000)
    colors = ('blue', 'green', 'darkorange', 'cyan', 'magenta')

    for i, seg_appr in enumerate(scorecards_dict.keys()):
        alpha = scorecards_dict[seg_appr]["Global Metrics"]["SPP"]["alpha"]
        beta = scorecards_dict[seg_appr]["Global Metrics"]["SPP"]["beta"]
        perf_mean = scorecards_dict[seg_appr]["Global Metrics"]["SPP"]["performance-mean"]

        fs = beta_distribution_pdf(xs, alpha, beta)
        plt.plot(xs, fs, 
                 linestyle='-', color=colors[i], label=SEG_DISPLAY_NAMES[seg_appr])

        plt.plot([perf_mean, perf_mean], [0, beta_distribution_pdf(perf_mean, alpha, beta)], 
                  linestyle='--', color=colors[i])

    plt.xlabel("Model Performance (Volume Overlap)")
    plt.ylabel("Probability Density Function")
    plt.legend()
    plt.title("Statistical Performance Profiles")

    if savefig:   
        plt.savefig(f"{output_dir}/combined_SPP_plot.png")
    else:
        plt.show()


def plot_combined_patient_dice(patient_ids, patient_metrics_dict, savefig, output_dir):
    colors = ('blue', 'green', 'darkorange', 'cyan', 'magenta')

    fig, ax = plt.subplots(figsize=(25,5))
    

    for i, seg_appr in enumerate(patient_metrics_dict.keys()):
        patient_dice_scores = patient_metrics_dict[seg_appr]['Dice']

        ax.plot(patient_ids, patient_dice_scores, 
                 linestyle='-', marker='o', color=colors[i], label=seg_appr)
        
        ax.set_yticks(np.arange(0, 1, 0.1))

    plt.xticks(fontsize=7, rotation=90)
    plt.ylabel("Dice Score")
    plt.grid()
    plt.legend()
    plt.title("Patient vs. Dice plot")
    plt.grid(axis='x')
    fig.tight_layout()
    if savefig:   
        plt.savefig(f"{output_dir}/combined_patient_dice_plot.png")
    else:
        plt.show()


def boxplot_patient_metrics(patient_ids, patient_metrics_dict, metric_name, savefig, output_dir):
    fig, ax = plt.subplots()

    seg_apporaches = patient_metrics_dict.keys()
    label_names = [SEG_DISPLAY_NAMES[seg_appr] for seg_appr in seg_apporaches]

    if metric_name == 'dice':
        vector_sequence = [np.array(patient_metrics_dict[seg_appr]['Dice']) for seg_appr in seg_apporaches]

    elif metric_name == 'hausdorff':
        vector_sequence = []
        for seg_appr in seg_apporaches:
            hausdorff_distances = patient_metrics_dict[seg_appr]['Hausdorff']
            hausdorff_distances = [hd if not math.isnan(hd) else 9999 for hd in hausdorff_distances] # NaN handling
            vector_sequence.append(np.array(hausdorff_distances))
            
    ax.boxplot(vector_sequence, labels=label_names, 
               showfliers=True,
               sym='rx')
    
    plt.xticks(rotation=20)
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} distributions")
    fig.tight_layout()
    
    if savefig:   
        plt.savefig(f"{output_dir}/boxplot_patient_{metric_name}.png")
    else:
        plt.show()




if __name__ == '__main__':
    main()