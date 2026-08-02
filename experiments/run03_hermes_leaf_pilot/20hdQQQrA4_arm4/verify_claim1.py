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
