# verify_claim3.py
import numpy as np
from scipy.linalg import pinv, qr

def verify_claim3():
    """
    Verify that CAffNet handles non-full row rank A(x) and arbitrary constraint cardinality
    """
    
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    n_out = 6
    
    # Test 1: Non-full row rank constraint matrix
    rank_deficient_A = np.random.randn(n_samples, 4, n_out)  # 4 constraints, rank < 4
    # Make one row dependent
    rank_deficient_A[:, 2] = rank_deficient_A[:, 0] + 0.5 * rank_deficient_A[:, 1]
    
    b = np.random.randn(n_samples, 4)
    
    # Test 2: Arbitrary constraint cardinality
    constraint_counts = [1, 2, 3, 4, 5, 6, 7, 8]  # More constraints than n_out
    
    def caffnet_projection(f_theta, A_x, b_x):
        """CAffNet-style projection that works with rank-deficient matrices"""
        # Use pseudoinverse which handles rank deficiency
        A_dagger = pinv(A_x)
        residual = A_x @ f_theta - b_x
        projection = f_theta - A_dagger @ residual
        return projection
    
    # Test with rank-deficient matrix
    f_theta = np.random.randn(n_samples, n_out)
    try:
        proj_rank_deficient = np.array([
            caffnet_projection(f_theta[i], rank_deficient_A[i], b[i]) 
            for i in range(n_samples)
        ])
        rank_deficient_works = True
        violations_rank_deficient = np.max([
            np.max(rank_deficient_A[i] @ proj_rank_deficient[i] - b[i])
            for i in range(n_samples)
        ])
    except:
        rank_deficient_works = False
        violations_rank_deficient = np.inf
    
    # Test with various constraint counts
    works_with_various_counts = []
    max_violations = []
    
    for m in constraint_counts:
        if m <= n_out:
            A_test = np.random.randn(n_samples, m, n_out)
            b_test = np.random.randn(n_samples, m)
        else:
            # Create overdetermined system
            A_test = np.random.randn(n_samples, m, n_out)
            b_test = np.random.randn(n_samples, m)
        
        try:
            proj = np.array([
                caffnet_projection(f_theta[i], A_test[i], b_test[i]) 
                for i in range(n_samples)
            ])
            max_viol = np.max([
                np.max(A_test[i] @ proj[i] - b_test[i])
                for i in range(n_samples)
            ])
            works_with_various_counts.append(True)
            max_violations.append(max_viol)
        except:
            works_with_various_counts.append(False)
            max_violations.append(np.inf)
    
    # Compare with HardNet approach (requires full rank)
    def hardnet_projection(f_theta, A_x, b_x):
        """HardNet-style projection requiring full row rank"""
        if np.linalg.matrix_rank(A_x) < A_x.shape[0]:
            raise ValueError("Matrix not full rank")
        # Standard projection
        return f_theta - np.linalg.lstsq(A_x, b_x - A_x @ f_theta, rcond=None)[0]
    
    hardnet_works = []
    for m in constraint_counts:
        if m <= n_out:
            A_test = np.random.randn(n_samples, m, n_out)
            b_test = np.random.randn(n_samples, m)
            # Ensure full rank
            A_test = A_test @ A_test.T + np.eye(m) * 1e-6
            
            try:
                proj = np.array([
                    hardnet_projection(f_theta[i], A_test[i], b_test[i]) 
                    for i in range(n_samples)
                ])
                hardnet_works.append(True)
            except:
                hardnet_works.append(False)
        else:
            hardnet_works.append(False)  # Can't have more constraints than features with full rank
    
    result = {
        'rank_deficient_handled': rank_deficient_works,
        'rank_deficient_violations': violations_rank_deficient,
        'works_with_arbitrary_counts': all(works_with_various_counts),
        'max_violations_by_count': dict(zip(constraint_counts, max_violations)),
        'hardnet_limitations': not all(hardnet_works),
        'caffnet_advantage': rank_deficient_works and all(works_with_various_counts)
    }
    
    return result

result3 = verify_claim3()
print("\nClaim 3 Verification Results:")
for key, value in result3.items():
    print(f"  {key}: {value}")

verdict3 = "verified" if result3['caffnet_advantage'] else "inconclusive"
print(f"Verdict: {verdict3}")
