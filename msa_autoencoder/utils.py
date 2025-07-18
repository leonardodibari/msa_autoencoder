import numpy as np
from scipy.spatial import distance
import tqdm
from Bio import SeqIO


def step_distances(x_traj, hamming=False) : 

    """
    Calculate traveled distances between two steps
    """

    d = []

    if hamming :

        for i in range(1, x_traj.shape[0]) : 
            d.append(distance.hamming(x_traj[i], x_traj[i-1]))
    else : 
        for i in range(1, x_traj.shape[0]) : 

            d.append(np.linalg.norm(x_traj[i] - x_traj[i-1]))

    return np.array(d)



def dist_traveled(x_traj, hamming=False) : 

    """
    Calculate the distances to the first point
    """

    x0 = x_traj[0]
    d = np.zeros(x_traj.shape[0])

    if hamming :
        for i in range(x_traj.shape[0]) :
            
            d[i] = distance.hamming(x_traj[i], x0)

    else : 

        for i in range(x_traj.shape[0]) : 

            d[i] = np.linalg.norm(x_traj[i] - x0)

    return d

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def dist_profile(x_traj) : 

    d = np.abs(x_traj - x_traj[0])
    return d[1:].mean(axis=0)

def grad_profile(x_traj, trained_GM) : 

    grad_mean = np.zeros(x_traj.shape[1])

    for i in tqdm.tqdm(range(x_traj.shape[0])) : 

        grad_mean += trained_GM.grad(x_traj[i])

    return grad_mean / x_traj.shape[0]


def read_fasta(file_path):
    sequences = [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]
    return sequences

def one_hot_encode(sequences, encoder, alphabet="ACDEFGHIKLMNPQRSTVWY-"):
    """ Encodes sequences into a binary matrix (num_sequences, seq_length * alphabet_size). """
    
    # Convert sequences into a list of character lists
    sequences_list = [list(seq) for seq in sequences]

    # Fit OneHotEncoder on the alphabet
    #encoder = OneHotEncoder(categories=[list(alphabet)], sparse_output=False, handle_unknown='ignore')
    encoded = encoder.transform(np.array(sequences_list).reshape(-1, 1))

    # Reshape to (num_sequences, seq_length * alphabet_size)
    encoded = encoded.reshape(len(sequences), len(sequences[0]) * len(alphabet))
    
    return encoded

def DMS(seq,) : 

    """
    Deep Mutation scanning
    """

    res = np.zeros((len(seq) * 20 +1, len(seq)))
    res[0] = seq
 
    for i, aa in enumerate(seq) :
        k = 0
        new_seq = seq.copy()
        for j in range(21) : 
            
            if aa != j :
                new_seq[i] = j                 
                res[i * 20 + k + 1] = new_seq 
                k += 1
    return res

# Okhubo's method for drift and diffusion estimation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gaussian_kernel(x, x0, W):

    return 1 / (np.sqrt(2 * np.pi ) * W) * np.exp(-(x - x0)**2 / (2 * W**2))


def drift_diff_est(traj, x0, W, dt = 1) : 

    Dtraj = traj[1:] - traj[:-1]
    KW = gaussian_kernel(traj, x0 , W)[:-1]
    KW_sum = dt * np.sum(KW, axis=0)

    mu = np.sum(Dtraj * KW, axis=0) / KW_sum

    sigma2 = np.sum((Dtraj - mu * dt)**2 * KW, axis=0) / KW_sum

    return mu, sigma2