import torch

def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    selected_indices = []

    for _ in range(K):
        # Select the index with the highest score
        idx_max = torch.argmax(scores).item()
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to -inf
        # to ensure it's not selected again
        scores[idx_max] = -float('inf')

    return selected_indices




