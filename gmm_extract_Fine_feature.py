import numpy as np
from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import argparse
import os

parser = argparse.ArgumentParser(description="Load and process dataset") 
parser.add_argument("--dataset", type=str,required=True, help="Dataset to use (reRLDD or reDROZY)") 
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()
np.random.seed(args.seed)
path="./confident_analysis"
save_path=f"./fine_grained_feature1/{args.dataset}"
if not os.path.exists(save_path): 
        os.makedirs(save_path) 
        print(f"Save path '{save_path}' created.") 
else: 
        print(f"Save path '{save_path}' already exists.")
if args.dataset=="reRLDD":
    c0= 3
    c1= 5
    n_components_0=4
    n_components_1=2
elif args.dataset=="reDRZOY":
    c0= 4
    c1= 6
    n_components_0=1
    n_components_1=1
else:
    print(f"Error: Unsupported dataset '{args.dataset}'. Please provide a valid dataset.") 

clean0 = np.load(f"{path}/{args.dataset}/clean0_data.npy").reshape(-1,8)
clean1 = np.load(f"{path}/{args.dataset}/clean1_data.npy").reshape(-1,8)
kmeans0 = KMeans(n_clusters=c0, init='k-means++', random_state=42)
labels0 = kmeans0.fit_predict(clean0)
kmeans1 = KMeans(n_clusters=c1, init='k-means++', random_state=42)
labels1 = kmeans1.fit_predict(clean1)
gmm_models0 = {i: GaussianMixture(n_components=n_components_0, random_state=42).fit(clean0[labels0 == i]) for i in range(c0)}
gmm_models1 = {i: GaussianMixture(n_components=n_components_1, random_state=42).fit(clean1[labels1 == i]) for i in range(c1)}

alphas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]  
combined_weights = []
combined_means = []
combined_covariances = []

for i in range(c0):
    gmm0 = gmm_models0[i]
    max_weight_index_A = np.argmax(gmm0.weights_)
    weight0, mean0, cov0 = gmm0.weights_[max_weight_index_A], gmm0.means_[max_weight_index_A], gmm0.covariances_[max_weight_index_A]
    for j in range(c1):
        gmm1 = gmm_models1[j] 
        max_weight_index_B = np.argmax(gmm1.weights_)
        weight1, mean1, cov1 = gmm1.weights_[max_weight_index_B], gmm1.means_[max_weight_index_B], gmm1.covariances_[max_weight_index_B]
        for alpha in alphas:
            beta = 1 - alpha
            combined_weights.append((beta, alpha))
            combined_means.append((mean0, mean1))
            combined_covariances.append((cov0, cov1))
combined_weights = np.array(combined_weights)
combined_means = np.array(combined_means)
combined_covariances = np.array(combined_covariances)

print("Combined weights:", combined_weights.shape)
print("Combined means:", combined_means.shape)
print("Combined covariances:", combined_covariances.shape)

remaining0= np.load(f"{path}/{args.dataset}/normalize_train.npy").reshape(-1,8)  
remaining1 = np.load(f"{path}/{args.dataset}/normalize_test.npy").reshape(-1,8)  

def process_samples(samples, alphas, component_means, component_covariances, component_weights):
    log_probs = np.zeros((samples.shape[0], len(component_means)))
    for idx, (means, covariances, weights) in enumerate(
            zip(component_means, component_covariances, component_weights)):
        mean0, mean1 = means
        cov0, cov1 = covariances
        beta, alpha = weights
        combined_mean = beta * mean0 + alpha * mean1
        combined_cov = beta*beta*cov0 + alpha*alpha*cov1
        combined_pdf = multivariate_normal(mean=combined_mean, cov=combined_cov).pdf(samples)
        log_probs[:, idx] = np.log(combined_pdf + 1e-10)
    best_indices = np.argmax(log_probs, axis=1)
    best_log_probs = log_probs[np.arange(samples.shape[0]), best_indices]
    result = [alphas[i % len(alphas)] for i in best_indices]
    count = Counter(result)
    total = len(result)
    percentages = {k: v / total for k, v in count.items()}

    return count, percentages, reshaped_result
count0, percentages0, retrain = process_samples(remaining0, alphas, combined_means, combined_covariances,combined_weights)
greater_than_0_5_count0 = sum(v for k, v in count0.items() if k < 0.5)
greater_than_or_equal_0_5_count0 = sum(v for k, v in count0.items() if k <= 0.5)
between_0_and_1_count0 = sum(v for k, v in count0.items() if 0 < k < 1)
count0_5 = sum(v for k, v in count0.items() if k == 0.5)

greater_than_0_5_percentage0 = greater_than_0_5_count0 / len(remaining0)
greater_than_or_equal_0_5_percentage0 = greater_than_or_equal_0_5_count0 / len(remaining0)
between_0_and_1_percentage0 = between_0_and_1_count0 / len(remaining0)
between0_5 = count0_5 / len(remaining0)

count1, percentages1, redattest = process_samples(remaining1, alphas, combined_means, combined_covariances,combined_weights)
less_than_0_5_count1 = sum(v for k, v in count1.items() if k > 0.5)
less_than_or_equal_0_5_count1 = sum(v for k, v in count1.items() if k >= 0.5)
between_0_and_1_count1 = sum(v for k, v in count1.items() if 0 < k < 1)
count1_5 = sum(v for k, v in count1.items() if k == 0.5)

less_than_0_5_percentage1 = less_than_0_5_count1 / len(remaining1)
less_than_or_equal_0_5_percentage1 = less_than_or_equal_0_5_count1 / len(remaining1)
between_0_and_1_percentage1 = between_0_and_1_count1 / len(remaining1)
between1_5 = count1_5 / len(remaining1)
np.save(f'{save_path}/train.npy', retrain)
np.save(f'{save_path}/test.npy', redattest)

print("处理 remaining0 的结果:")
print("计数:", count0)
print("百分比:", percentages0)
print("小于 0.5 的 alphas 百分比:", greater_than_0_5_percentage0)
print("小于等于 0.5 的 alphas 百分比:", greater_than_or_equal_0_5_percentage0)
print("等于 0.5 的 alphas 百分比:", between0_5)
print("在 0 和 1 之间（不包括）的 alphas 百分比:", between_0_and_1_percentage0)

print("\n处理 remaining1 的结果:")
print("计数:", count1)
print("百分比:", percentages1)
print("大于 0.5 的 alphas 百分比:", less_than_0_5_percentage1)
print("大于等于 0.5 的 alphas 百分比:", less_than_or_equal_0_5_percentage1)
print("等于 0.5 的 alphas 百分比:", between1_5)
print("在 0 和 1 之间（不包括）的 alphas 百分比:", between_0_and_1_percentage1)



