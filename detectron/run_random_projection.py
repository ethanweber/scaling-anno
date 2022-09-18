"""Create the random matrices and somewhere.
"""
import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def save_matrix(filename, input_dim, output_dim):
    X = np.random.rand(1, input_dim)
    transformer = GaussianRandomProjection(n_components=output_dim, random_state=0)
    fitted = transformer.fit(X)
    matrix = fitted.components_.T
    np.save(filename, matrix)


# save_matrix("../detectron/projection_matrices/features_backbone.npy", 256 * 7 * 7, 1024)
# save_matrix("../detectron/projection_matrices/features_logits_fine.npy", 56 * 56, 1024)
# save_matrix("../detectron/projection_matrices/features_attention_1024.npy", 256 * 7 * 7, 1024)
# save_matrix("../detectron/projection_matrices/features_attention_512.npy", 256 * 7 * 7, 512)
# save_matrix("../detectron/projection_matrices/features_attention_256.npy", 256 * 7 * 7, 256)

# save_matrix("../detectron/projection_matrices/features_backbone_512.npy", 256 * 7 * 7, 512)
# save_matrix("../detectron/projection_matrices/features_mask_512.npy", 56 * 56, 512)
