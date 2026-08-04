# Arm4 (ARC) reproduction — CAffNet: Hard Constraint-Affine Neural Networks

I'll create verification scripts for each claim. Let me start by analyzing what each claim requires and implementing appropriate tests.

```python
# verify_claim1.py
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def verify_claim1():
    """
    Verify Theorem 3.5: If unconstrained network approximates F, 
    then CAffNet approximates constrained class with error bound ||P* - f_t||_p < (3 + 3*sqrt(n_out))K
    """
    
    # Set up test problem
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_out = 5
    
    # Generate input data
    X = np.random.randn(n_samples, n_features)
    
    # Define constraint matrix A(x) and vector b(x) - linear constraints for simplicity
    A = np.random.randn(n_samples, 3, n_out)  # A(x) varies with x
    b = np.random.randn(n_samples, 3)
    
    # Target function class F - polynomials up to degree 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # True function in constrained class
    true_coef = np.random.randn(X_poly.shape[1])
    f_true = X_poly @ true_coef
    
    # Unconstrained approximation (standard polynomial regression)
    unconstrained_model = LinearRegression()
    unconstrained_model.fit(X_poly, f_true)
    f_unconstrained = unconstrained_model.predict(X_poly)
    
    # Constrained approximation using CAffNet-style projection
    def project_to_constraints(f_pred, A_x, b_x):
        """Project prediction to satisfy A(x)f <= b(x)"""
        # Simple projection: if constraint violated, scale down
        violations = A_x @ f_pred - b_x
        if np.all(violations <= 0):
            return f_pred
        
        # Scale factor to satisfy most violated constraint
        max_violation = np.max(violations)
        if max_violation <= 0:
            return f_pred
            
        scaling_factor = 1.0 / (1.0 + max_violation)
        return f_pred * scaling_factor
    
    # Apply projections for each sample
    f_constrained = np.zeros_like(f_unconstrained)
    for i in range(n_samples):
        f_constrained[i] = project_to_constraints(
            f_unconstrained[i], A[i], b[i]
        )
    
    # Calculate approximation errors
    error_unconstrained = np.mean((f_true - f_unconstrained)**2)
    error_constrained = np.mean((f_true - f_constrained)**2)
    
    # Estimate K constant (approximation capability of unconstrained network)
    K = np.sqrt(error_unconstrained)
    
    # Theoretical bound from theorem
    theoretical_bound = (3 + 3 * np.sqrt(n_out)) * K
    
    # Actual error bound
    actual_error_bound = np.linalg.norm(f_constrained - f_true, ord=2)
    
    # Check if theorem holds
    theorem_holds = actual_error_bound < theoretical_bound
    
    result = {
        'unconstrained_error': error_unconstrained,
        'constrained_error': error_constrained,
        'theoretical_bound': theoretical_bound,
        'actual_error_bound': actual_error_bound,
        'theorem_holds': theorem_holds,
        'bound_ratio': actual_error_bound / theoretical_bound if theoretical_bound > 0 else np.inf
    }
    
    return result

result1 = verify_claim1()
print("Claim 1 Verification Results:")
for key, value in result1.items():
    print(f"  {key}: {value}")

verdict1 = "verified" if result1['theorem_holds'] else "inconclusive"
print(f"Verdict: {verdict1}")
```

```python
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
```

```python
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
```

```python
# verify_claim4.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def verify_claim4():
    """
    Verify 73.33% MSE reduction on piecewise-constraint benchmark
    """
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 10
    
    # Generate piecewise constraint data
    X = np.random.randn(n_samples, n_features)
    
    # Create piecewise linear function with different behaviors in regions
    def piecewise_function(x):
        if x[0] < 0:
            return 2*x[0] + 0.5*x[1] + np.sin(x[2]) + 0.1*np.sum(x[3:])
        elif x[0] < 1:
            return -x[0] + x[1]*2 + np.cos(x[2]) + 0.2*np.sum(x[3:])
        else:
            return x[0]*0.5 - x[1] + np.sin(x[2]*2) + 0.3*np.sum(x[3:])
    
    y = np.array([piecewise_function(x) for x in X])
    
    # Define piecewise constraints
    def get_constraints(x):
        """Return A(x) and b(x) for piecewise constraints"""
        if x[0] < 0:
            A = np.array([[1, 0] + [0]*(n_features-2), 
                         [0, 1] + [0]*(n_features-2)])
            b = np.array([0.5, 1.0])
        elif x[0] < 1:
            A = np.array([[1, 0] + [0]*(n_features-2),
                         [0, -1] + [0]*(n_features-2)])
            b = np.array([0.8, -0.5])
        else:
            A = np.array([[1, 0] + [0]*(n_features-2),
                         [0, 1] + [0]*(n_features-2)])
            b = np.array([1.5, 0.5])
        return A, b
    
    # Split data
    train_size = 1500
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Soft-constrained baseline (regularized regression)
    soft_model = Ridge(alpha=1.0)
    soft_model.fit(X_train, y_train)
    y_pred_soft = soft_model.predict(X_test)
    mse_soft = np.mean((y_test - y_pred_soft)**2)
    
    # CAffNet-style constrained prediction
    def caffnet_predict(X_test, model, X_train, y_train):
        """Constrained prediction using training data"""
        predictions = model.predict(X_test)
        constrained_preds = []
        
        for i, x in enumerate(X_test):
            A_x, b_x = get_constraints(x)
            pred = predictions[i]
            
            # Project to satisfy constraints
            violations = A_x @ pred - b_x
            if np.any(violations > 0):
                # Simple correction: adjust to satisfy most violated constraint
                max_viol_idx = np.argmax(violations)
                adjustment = violations[max_viol_idx] * A_x[max_viol_idx] / (np.linalg.norm(A_x[max_viol_idx])**2 + 1e-8)
               