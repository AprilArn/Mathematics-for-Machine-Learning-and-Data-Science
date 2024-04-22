import math
import numpy as np
import pandas as pd
from scipy import stats

# Load the data from the test using pd.read_csv
data = pd.read_csv("background_color_experiment.csv")

# Print the first 10 rows
data.head(10)

print(f"The dataset size is: {len(data)}")

# Separate the data from the two groups (sd stands for session duration)
control_sd_data = data[data["user_type"]=="control"]["session_duration"]
variation_sd_data = data[data["user_type"]=="variation"]["session_duration"]

print(f"{len(control_sd_data)} users saw the original website with an average duration of {control_sd_data.mean():.2f} minutes\n")
print(f"{len(variation_sd_data)} users saw the new website with an average duration of {variation_sd_data.mean():.2f} minutes")

# X_c stores the session tome for the control group and X_v, for the variation group. 
X_c = control_sd_data.to_numpy()
X_v = variation_sd_data.to_numpy()

print(f"The first 10 entries for X_c are:\n{X_c[:20]}\n")
print(f"The first 10 entries for X_v are:\n{X_v[:20]}\n")


def get_stats(X):
    """
    Calculate basic statistics of a given data set.

    Parameters:
    X (numpy.array): Input data.

    Returns:
    tuple: A tuple containing:
        - n (int): Number of elements in the data set.
        - x (float): Mean of the data set.
        - s (float): Sample standard deviation of the data set.
    """

    ### START CODE HERE ###
    
    # Get the group size
    n = len(X)
    # Get the group mean
    x = X.mean()
    # Get the group sample standard deviation (do not forget to pass the parameter ddof if using the method .std)
    s = X.std(ddof=1)

    ### END CODE HERE ###

    return (n,x,s)

n_c, x_c, s_c = get_stats(X_c)
n_v, x_v, s_v = get_stats(X_v)

print(f"For X_c:\n\tn_c = {n_c}, x_c = {x_c:.2f}, s_c = {s_c:.2f} ")
print(f"For X_v:\n\tn_v = {n_v}, x_v = {x_v:.2f}, s_v = {s_v:.2f} ")

def degrees_of_freedom(n_v, s_v, n_c, s_c):
    """Computes the degrees of freedom for two samples.

    Args:
        control_metrics (estimation_metrics_cont): The metrics for the control sample.
        variation_metrics (estimation_metrics_cont): The metrics for the variation sample.

    Returns:
        numpy.float: The degrees of freedom.
    """
    
    ### START CODE HERE ###
    
    # To make the code clean, let's divide the numerator and the denominator. 
    # Also, note that the value s_c^2/n_c and s_v^2/n_v appears both in the numerator and denominator, so let's also compute them separately

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = np.square(s_v) / n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = np.square(s_c) / n_c


    # Compute the numerator in the formula given above
    numerator = np.square(s_v_n_v + s_c_n_c)

    # Compute the denominator in the formula given above. Attention that s_c_n_c and s_v_n_v appears squared here!
    # Also, remember to use parenthesis to indicate the operation order. Note that a/b+1 is different from a/(b+1).
    denominator = np.square(s_c_n_c) / (n_c - 1) + np.square(s_v_n_v) / (n_v - 1)
    
    ### END CODE HERE ###

    dof = numerator/denominator
        
    return dof

d = degrees_of_freedom(n_v, s_v, n_c, s_c)
print(f"The degrees of freedom for the t-student in this scenario is: {d:.2f}")

def t_value(n_v, x_v, s_v, n_c, x_c, s_c):

    ### START CODE HERE ###

    # As you did before, let's split the numerator and denominator to make the code cleaner.
    # Also, let's compute again separately s_c^2/n_c and s_v^2/n_v.

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = np.square(s_v) / n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = np.square(s_c) / n_c


    # Compute the numerator for the t-value as given in the formula above
    numerator = x_v - x_c

    # Compute the denominator for the t-value as given in the formula above. You may use np.sqrt to compute the square root.
    denominator = np.sqrt(s_v_n_v + s_c_n_c)
    
    ### END CODE HERE ###

    t = numerator/denominator

    return t

t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)
print(f"The t-value for this experiment is: {t:.2f}")

t_10 = stats.t(df = 10)
cdf = t_10.cdf(1.21)
print(f"The CDF for the t-student distribution with 10 degrees of freedom and t-value = 1.21, or equivalently P(t_10 < 1.21) is equal to: {cdf:.2f}")

def p_value(d, t_value):

    ### START CODE HERE ###

    # Load the t-student distribution with $d$ degrees of freedom. Remember that the parameter in the stats.t is given by df.
    t_d = stats.t(df=d)

    # Compute the p-value, P(t_d > t). Remember to use the t_d.cdf with the proper adjustments as discussed above.
    p = 1 - t_d.cdf(t_value)


    ### END CODE HERE ###

    return p

print(f"The p-value for t_15 with t-value = 1.10 is: {p_value(15, 1.10):.4f}")
print(f"The p-value for t_30 with t-value = 1.10 is: {p_value(30, 1.10):.4f}")

