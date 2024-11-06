import re
import numpy as np
import pandas as pd
import matplotlib
import pickle

top30 =['Fe', 'Ni', 'Cr', 'Co', 'Al', 'Ti', 'Cu',
        'Mo', 'V', 'Nb', 'Mn', 'Zr', 'Ta', 'W', 'Hf',
        'Si', 'Rh', 'Ru', 'Re', 'Os', 'Pt', 'Pd', 'Ir',
        'Mg', 'Ag', 'Zn', 'Sn', 'Au', 'Li', 'Sc']

ftrs = ['tm', 'vac', 'vm',  'k', 'ar', 'chi', 'delta_s_mix', 'delta_h_mix']
ftrs_names = ['tm', 'vac', 'vm',  'k', 'delta', 'delta_chi', 'delta_s_mix', 'delta_h_mix']

with open('/home/hughes/usr/d_vae_hea/data/look_up_dict.pkl', 'rb') as pf:
    look_up_dict = pickle.load(pf)
with open('/home/hughes/usr/d_vae_hea/data/mixing_enthalpy_dict.pkl', 'rb') as pf:
    mixing_enthalpy_dict = pickle.load(pf)

def vectorize_alloy(alloy):
    pattern = re.compile(r'([A-Z][a-z]*)(\d*\.*\d*?(?=\D|$))')
    alloy_sep = pattern.findall(alloy)
    alloy_sep = [(x, float(y)) if y else (x, 1) for x, y in alloy_sep]
    elements = [x for x, y in alloy_sep]
    comps = [y for x, y in alloy_sep]
    normalizer = sum(comps)
    comps = np.array(comps)/normalizer
    alloy_comps = {ele:comp/normalizer for ele, comp in alloy_sep}
    return alloy_comps, elements, comps

def calculate_compositions(alloy):
    # Input is the chemical formula for an alloy
    top30_comps = []
    alloy_comps, elements, comps = vectorize_alloy(alloy)
    if not set(elements).issubset(top30):
        raise NotImplementedError('Containing elements that are not implemented!')
    for ele in top30:
        if ele not in elements:
            top30_comps.append(0)
        else:
            top30_comps.append(100*alloy_comps[ele])
    sorted_alloy = ''.join([str(x)+str(int(y)) for x, y in zip(top30, top30_comps) if y > 0])
    return top30_comps, alloy_comps, sorted_alloy

def calculate_engineered_features(alloy, scaler=None):
    convert_indice = [3, 2, 0, 1, 4, 5, 6, 7]
    R = 8.314
    alloy_comps, elements, comps = vectorize_alloy(alloy)
    ftr_matrix = np.zeros((len(elements),len(ftrs))) # feature vector
    for i, (ele, comp) in enumerate(zip(elements, comps)):
        for j, ftr in enumerate(ftrs[:6]):
            if j != 4 and j !=5 :
                ftr_matrix[i, j] = look_up_dict[ele][ftr] * comp
            else:
                ftr_matrix[i, j] = look_up_dict[ele][ftr]
    for i, comp in enumerate(comps):
        ftr_matrix[i, 6] = - R * comp * np.log(comp + np.finfo(np.float32).eps)
    for i, (ele_a, comp_a) in enumerate(zip(elements, comps)):
        for j, (ele_b, comp_b) in enumerate(zip(elements, comps)):
            if ele_b != ele_a:
                combo_key = [ele_a, ele_b]
                combo_key.sort()
                combo_key = ','.join(combo_key)
                ftr_matrix[i,7] +=  4*comp_a * comp_b * mixing_enthalpy_dict[combo_key] / 2
    ftr_vec = np.zeros(len(ftrs))
    ftr_vec[:4] = np.sum(ftr_matrix[:, :4], axis=0)
    avg_ar, avg_chi = ftr_matrix[:, 4:6].T @ comps
    ftr_vec[4] = 100*np.sqrt((comps * (1 - ftr_matrix[:, 4]/avg_ar)**2).sum())
    ftr_vec[5] = np.sqrt((comps * (ftr_matrix[:, 5] - avg_chi)**2).sum())
    ftr_vec[6:] = np.sum(ftr_matrix[:, 6:], axis=0)
    ftr_vec = np.array(ftr_vec)[convert_indice]
    if scaler:
        scaler = np.load(scaler)
        ftr_vec = (ftr_vec - scaler[0])/scaler[1]
    return ftr_vec
