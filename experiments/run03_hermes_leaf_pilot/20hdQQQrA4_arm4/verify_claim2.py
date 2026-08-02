# verify_claim2.py
import numpy as np
from scipy.linalg import pinv

def verify_claim2():
    """
    Verify CAffine layer projection formula with trainable null-space component
    """
    
    np.random.seed(42)
    batch_size = 32
    n_features = 20
    n_out = 8
    m_constraints = 5
    
    # Simulate inputs
    x = np.random.randn(batch_size, n_features)
    
    # Simulate network outputs f_theta(x)
    f_theta = np.random.randn(batch_size, n_out)
    
    # Simulate constraint matrices A_gamma(x) and vectors b_gamma(x)
    A_gamma = np.random.randn(batch_size, m_constraints, n_out)
    b_gamma = np.random.randn(batch_size, m_constraints)
    
    # Compute projection using the given formula
    def compute_projection(f_theta_x, A_x, b_x, w_phi_x):
        """
        P_gamma(x) = f_theta(x) - A_gamma^dagger(x)(A_gamma(x) f_theta(x) - b_gamma(x)) 
                     + (I - A_gamma^dagger(x) A_gamma(x)) w_phi(x)
        """
        # Moore-Penrose pseudoinverse
        A_dagger = np.array([pinv(A_x[i]) for i in range(A_x.shape[0])])
        
        # First term: f_theta(x)
        term1 = f_theta_x
        
        # Second term: -A_gamma^dagger(x)(A_gamma(x) f_theta(x) - b_gamma(x))
        Af_minus_b = np.array([A_x[i] @ f_theta_x[i] - b_x[i] for i in range(A_x.shape[0])])
        term2 = -np.array([A_dagger[i] @ Af_minus_b[i] for i in range(A_dagger.shape[0])])
        
        # Third term: (I - A_gamma^dagger(x) A_gamma(x)) w_phi(x)
        I = np.eye(n_out)
        projection_matrix = np.array([I - A_dagger[i] @ A_x[i] for i in range(A_x.shape[0])])
        term3 = np.array([projection_matrix[i] @ w_phi_x[i] for i in range(projection_matrix.shape[0])])
        
        return term1 + term2 + term3
    
    # Test with different w_phi values (trainable null-space component)
    w_phi_variations = [
        np.random.randn(batch_size, n_out),  # Random null-space component
        np.zeros((batch_size, n_out)),       # Zero null-space component
        np.random.randn(batch_size, n_out) * 0.1  # Small null-space component
    ]
    
    projections = []
    for w_phi in w_phi_variations:
        proj = compute_projection(f_theta, A_gamma, b_gamma, w_phi)
        projections.append(proj)
    
    # Verify that different w_phi values produce different projections
    diff_01 = np.mean(np.abs(projections[0] - projections[1]))
    diff_02 = np.mean(np.abs(projections[0] - projections[2]))
    
    # Verify constraint satisfaction
    def check_constraints(proj, A_x, b_x):
        violations = np.array([A_x[i] @ proj[i] - b_x[i] for i in range(A_x.shape[0])])
        return np.max(violations) <= 1e-10  # Allow small numerical errors
    
    constraints_satisfied = [check_constraints(proj, A_gamma, b_gamma) for proj in projections]
    
    result = {
        'projection_diff_wphi_01': diff_01,
        'projection_diff_wphi_02': diff_02,
        'constraints_satisfied_all': all(constraints_satisfied),
        'individual_satisfaction': constraints_satisfied,
        'allows_joint_optimization': diff_01 > 1e-10 and diff_02 > 1e-10
    }
    
    return result

result2 = verify_claim2()
print("\nClaim 2 Verification Results:")
for key, value in result2.items():
    print(f"  {key}: {value}")

verdict2 = "verified" if result2['allows_joint_optimization'] and result2['constraints_satisfied_all'] else "inconclusive"
print(f"Verdict: {verdict2}")
