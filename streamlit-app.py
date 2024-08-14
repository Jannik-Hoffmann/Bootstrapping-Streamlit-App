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

st.title("WTF is Bootstrapping?!?: An Interactive Demonstration")

st.write("""
**WTF is Bootstrapping: An Interactive Exploration**

Ever found bootstrapping in statistics a bit elusive? You're not alone! This app is designed to turn that complexity into clarity with hands-on visualizations and simulations. Whether you’re new to the concept or looking to solidify your understanding, you’ll find this tool to be a game-changer.

**What’s Bootstrapping All About?**
Bootstrapping is like magic for statisticians. By resampling your data with replacement, you can get insights about your population even when you don’t know its underlying distribution. It’s perfect for when:

- The actual distribution of the data from which the sample is drawn is a mystery
- You’re working with a small sample size
- You need to calculate confidence intervals for complex statistics

**Why Give Bootstrapping a Try?**
With bootstrapping, you can:

- Unveil the sampling distribution of almost any statistic
- Build confidence intervals without relying on normality
- Test hypotheses when traditional methods fall short due to the problems described above

Dive into my interactive demo and see bootstrapping in action! Adjust the parameters and watch how the bootstrap distribution and confidence intervals shift. This will be updated in the future and just serves as my first experiment with Streamlit Apps.
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
