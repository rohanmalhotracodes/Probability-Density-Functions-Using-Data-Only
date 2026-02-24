# Learning Probability Density Functions Using Data Only (GAN-based PDF Estimation)

**Name:** Rohan Malhotra  
**Roll Number:** 102303437  

## Title
Learning Probability Density Functions using data only

## Dataset
- **Feature used:** NO₂ concentration as the feature `x`
- **Dataset source:** India Air Quality Data (Kaggle)

> Dataset link: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## Objective

To learn an unknown probability density function of a transformed random variable using a **Generative Adversarial Network (GAN)**, using **data samples only** (without assuming any analytical/parametric PDF form).

---

## Problem Statement Summary

We are given samples of NO₂ concentration (`x`).  
Each value is transformed into `z` using:

`z = x + a_r * sin(b_r * x)`

where:
- `a_r = 0.5 * (r mod 7)`
- `b_r = 0.3 * ((r mod 5) + 1)`
- `r` is the university roll number

The goal is to train a GAN on samples of `z` so that the generator learns the distribution of `z`, and then estimate the PDF from generated samples.

---

## Transformation Parameters

Given:

- `a_r = 0.5 * (r mod 7)`
- `b_r = 0.3 * ((r mod 5) + 1)`

For `r = 102303437`:

- `r mod 7 = 5`
- `r mod 5 = 2`

Therefore:

- `a_r = 0.5 * 5 = 2.5`
- `b_r = 0.3 * (2 + 1) = 0.9`

### Transformation used

`z = x + a_r * sin(b_r * x)`

So for this assignment:

`z = x + 2.5 * sin(0.9 * x)`

---

## GAN Architecture Description

### Generator (MLP)

- **Input:** latent noise vector `e ~ N(0,1)`, dimension = `8`
- **Layers:**
  - `Linear(8 -> 32)` + `LeakyReLU`
  - `Linear(32 -> 64)` + `LeakyReLU`
  - `Linear(64 -> 32)` + `LeakyReLU`
  - `Linear(32 -> 1)`
- **Output:** scalar generated sample (normalized `z_f`)

### Discriminator (MLP)

- **Input:** scalar sample (`z` or `z_f`)
- **Layers:**
  - `Linear(1 -> 32)` + `LeakyReLU`
  - `Linear(32 -> 64)` + `LeakyReLU`
  - `Linear(64 -> 32)` + `LeakyReLU`
  - `Linear(32 -> 1)` + `Sigmoid`
- **Output:** probability that the sample is real

---

## Training Setup

- **Loss:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam
- **Noise input to generator:** Gaussian noise `N(0,1)`
- **Stability trick:** Real label smoothing used for improved training stability
- **Important:** GAN is trained **only on transformed samples `z`** (no parametric PDF assumption such as Gaussian/exponential)

---

## PDF Approximation from Generator Samples

After GAN training:

1. Generate a large number of samples from the generator (`z_f`)
2. Approximate `p_h(z)` using:
   - **Histogram density estimation**
   - **Kernel Density Estimation (KDE)**

---

## Repository Structure

- `assignment_gan_pdf_no2.ipynb` → Main runnable notebook
- `README.md` → Assignment description, method, and results summary

---

## How to Run

1. Download the dataset from Kaggle.
2. Open `assignment_gan_pdf_no2.ipynb`.
3. Update the dataset file path in the notebook if required.
4. Run all cells in order.
5. The notebook will:
   - Load and clean NO₂ data
   - Apply transformation to obtain `z`
   - Train the GAN on `z`
   - Generate samples from the trained generator
   - Plot histogram/KDE to approximate the learned PDF
