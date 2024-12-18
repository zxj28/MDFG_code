import numpy as np
from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import argparse
import os

parser = argparse.ArgumentParser(description="Load and process dataset") 
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (reRLDD or reDROZY)") 
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()
np.random.seed(args.seed)

path = "./confident_analysis"
save_path = f"./fine_grained_feature_zhushi1/{args.dataset}"

if not os.path.exists(save_path): 
    os.makedirs(save_path) 
    print(f"Save path '{save_path}' created.") 
else: 
    print(f"Save path '{save_path}' already exists.")

if args.dataset == "reRLDD":
    c0 = 3
    c1 = 5
    n_components_0 = 4
    n_components_1 = 2
elif args.dataset == "reDROZY":
    c0 = 4
    c1 = 6
    n_components_0 = 1
    n_components_1 = 1
else:
    print(f"Error: Unsupported dataset '{args.dataset}'. Please provide a valid dataset.") 

clean0 = np.load(f"{path}/{args.dataset}/clean0_data.npy").reshape(-1, 8)
clean1 = np.load(f"{path}/{args.dataset}/clean1_data.npy").reshape(-1, 8)

# Train KMeans and get cluster labels
kmeans0 = KMeans(n_clusters=c0, init='k-means++', random_state=42)
labels0 = kmeans0.fit_predict(clean0)
kmeans1 = KMeans(n_clusters=c1, init='k-means++', random_state=42)
labels1 = kmeans1.fit_predict(clean1)

gmm_models0 = {i: GaussianMixture(n_components=n_components_0, random_state=42).fit(clean0[labels0 == i]) for i in range(c0)}
gmm_models1 = {i: GaussianMixture(n_components=n_components_1, random_state=42).fit(clean1[labels1 == i]) for i in range(c1)}

# Define proportions
alphas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

# Parameters for the combined distributions
combined_weights = []
combined_means = []
combined_covariances = []

# Combine the first Gaussian components of each GMM using different proportions
for i in range(c0):
    gmm0 = gmm_models0[i]
    # Find the component with the largest weight
    max_weight_index_A = np.argmax(gmm0.weights_)
    # Extract parameters of the component with the largest weight
    weight0, mean0, cov0 = gmm0.weights_[max_weight_index_A], gmm0.means_[max_weight_index_A], gmm0.covariances_[max_weight_index_A]
    for j in range(c1):
        gmm1 = gmm_models1[j] 
        # Find the component with the largest weight
        max_weight_index_B = np.argmax(gmm1.weights_)
        # Extract parameters of the first Gaussian component
        weight1, mean1, cov1 = gmm1.weights_[max_weight_index_B], gmm1.means_[max_weight_index_B], gmm1.covariances_[max_weight_index_B]
        # Combine all possible proportions
        for alpha in alphas:
            beta = 1 - alpha
            combined_weights.append((beta, alpha))
            combined_means.append((mean0, mean1))
            combined_covariances.append((cov0, cov1))

# Convert to NumPy arrays
combined_weights = np.array(combined_weights)
combined_means = np.array(combined_means)
combined_covariances = np.array(combined_covariances)

print("Combined weights:", combined_weights.shape)
print("Combined means:", combined_means.shape)
print("Combined covariances:", combined_covariances.shape)

remaining0 = np.load(f"{path}/{args.dataset}/normalize_train.npy").reshape(-1, 8)  
remaining1 = np.load(f"{path}/{args.dataset}/normalize_test.npy").reshape(-1, 8)  

def process_samples(samples, alphas, component_means, component_covariances, component_weights):
    # Initialize log probability array
    log_probs = np.zeros((samples.shape[0], len(component_means)))

    for idx, (means, covariances, weights) in enumerate(
            zip(component_means, component_covariances, component_weights)):
        mean0, mean1 = means
        cov0, cov1 = covariances
        beta, alpha = weights

        # Compute the combined distribution mean
        combined_mean = beta * mean0 + alpha * mean1
        combined_cov = beta * beta * cov0 + alpha * alpha * cov1

        # Compute the probability density of the combined distribution
        combined_pdf = multivariate_normal(mean=combined_mean, cov=combined_cov).pdf(samples)

        # Compute log probability density (add a small constant to avoid log(0))
        log_probs[:, idx] = np.log(combined_pdf + 1e-10)

    # Find the combination with the highest log probability density for each sample
    best_indices = np.argmax(log_probs, axis=1)
    best_log_probs = log_probs[np.arange(samples.shape[0]), best_indices]

    # Compute alpha values corresponding to each sample
    result = [alphas[i % len(alphas)] for i in best_indices]
    reshaped_result = result

    # Count and calculate percentages for each alpha
    count = Counter(result)
    total = len(result)
    percentages = {k: v / total for k, v in count.items()}

    return count, percentages, reshaped_result

count0, percentages0, retrain = process_samples(remaining0, alphas, combined_means, combined_covariances, combined_weights)
count1, percentages1, redattest = process_samples(remaining1, alphas, combined_means, combined_covariances, combined_weights)
# Save results
np.save(f'{save_path}/train.npy', retrain)
np.save(f'{save_path}/test.npy', redattest)
