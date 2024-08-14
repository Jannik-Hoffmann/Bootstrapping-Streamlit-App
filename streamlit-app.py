import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(n, correlation):
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def bootstrap_correlation(data, n_iterations):
    correlations = []
    for _ in range(n_iterations):
        sample = resample(data)
        correlations.append(np.corrcoef(sample.T)[0, 1])
    return correlations

def resample(data):
    return data[np.random.randint(len(data), size=len(data))]

st.title("Bootstrapping in Statistics: An Interactive Demonstration")

st.write("""
## Purpose of this App
This app demonstrates the concept of bootstrapping in statistics through interactive visualizations and simulations.

## What is Bootstrapping?
Bootstrapping is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement. It's particularly useful when:
1. The underlying distribution of the data is unknown
2. The sample size is small
3. You want to calculate confidence intervals for complex statistics

## Why Use Bootstrapping?
Bootstrapping allows us to:
1. Estimate the sampling distribution of almost any statistic
2. Construct confidence intervals without assuming normality
3. Test hypotheses when traditional methods are not applicable

Let's explore bootstrapping through an interactive example!
""")

# Input parameters
st.sidebar.header("Data Generation Parameters")
n_samples = st.sidebar.slider("Sample Size", 10, 500, 100)
correlation = st.sidebar.slider("True Correlation", -1.0, 1.0, 0.5, 0.1)
n_bootstrap = st.sidebar.slider("Number of Bootstrap Samples", 100, 5000, 1000)

# Generate data
data = generate_data(n_samples, correlation)

# Plot original data
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Original Data (n={n_samples}, r={correlation:.2f})")
st.pyplot(fig)

# Perform bootstrapping
bootstrap_correlations = bootstrap_correlation(data, n_bootstrap)

# Plot bootstrap distribution
fig, ax = plt.subplots()
sns.histplot(bootstrap_correlations, kde=True, ax=ax)
ax.axvline(correlation, color='r', linestyle='--', label='True Correlation')
ax.axvline(np.mean(bootstrap_correlations), color='g', linestyle='--', label='Bootstrap Mean')
ax.set_xlabel("Correlation")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap Distribution (n_iterations={n_bootstrap})")
ax.legend()
st.pyplot(fig)

# Calculate and display confidence interval
confidence_interval = np.percentile(bootstrap_correlations, [2.5, 97.5])
st.write(f"95% Confidence Interval for Correlation: ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})")

st.write("""
## Interpretation
The histogram above shows the distribution of correlations obtained through bootstrapping. 
The red dashed line represents the true correlation we set, while the green dashed line shows the mean of our bootstrap samples.
The 95% confidence interval gives us a range where we can be 95% confident that the true correlation lies.

Try adjusting the parameters in the sidebar to see how they affect the bootstrap distribution and confidence interval!
""")
