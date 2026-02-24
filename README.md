# Learning Probability Density Functions using Data Only (GAN)

**Student Name:** Rohan Malhotra  
**Roll Number:** 102303437  

## Assignment Title
**Learning Probability Density Functions using data only**

## Dataset
- **Feature:** NO₂ concentration (`x`)
- **Dataset source:** India Air Quality Data (Kaggle link provided in assignment)

## Transformation Parameters (from Roll Number)
Given:
- \(a_r = 0.5 (r \bmod 7)\)
- \(b_r = 0.3 ((r \bmod 5)+1)\)

For **r = 102303437**:
- \(r \bmod 7 = 5\)
- \(r \bmod 5 = 2\)

Therefore:
- **a_r = 2.5**
- **b_r = 0.9**

Transformation used:
\[
z = x + a_r \sin(b_r x)
\]

---

## GAN Architecture Description (1D GAN)

### Generator (MLP)
- Input: latent noise vector \(\epsilon \sim \mathcal{N}(0,1)\), dimension = 8
- Layers:
  - Linear(8 → 32) + LeakyReLU
  - Linear(32 → 64) + LeakyReLU
  - Linear(64 → 32) + LeakyReLU
  - Linear(32 → 1)
- Output: scalar generated sample (normalized `z_f`)

### Discriminator (MLP)
- Input: scalar sample (`z` or `z_f`)
- Layers:
  - Linear(1 → 32) + LeakyReLU
  - Linear(32 → 64) + LeakyReLU
  - Linear(64 → 32) + LeakyReLU
  - Linear(32 → 1) + Sigmoid
- Output: probability that sample is real

### Training Setup
- Loss: Binary Cross Entropy (BCE)
- Optimizer: Adam
- Real label smoothing used for better stability
- GAN is trained **only on transformed samples `z`** (no parametric PDF assumed)

---

## PDF Approximation from Generator Samples
After GAN training:
1. Generate a large number of samples from the generator (`z_f`)
2. Approximate \(p_h(z)\) using:
   - Histogram density estimation
   - Kernel Density Estimation (KDE)

---

## Project Structure
```text
rohan-malhotra-gan-pdf-assignment/
├── README.md
└── assignment_gan_pdf_no2.ipynb
```

---

## How to Run
1. Download the Kaggle dataset from the assignment link.
2. Place the CSV file locally.
3. Open `assignment_gan_pdf_no2.ipynb`.
4. Update `CSV_PATH` in the notebook.
5. Run all cells.

### Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn torch
```

---

## What to Send Me After Running (for README update)
Please share:
- Final PDF plot (real vs generated + KDE)
- Training loss plot
- Any printed metrics / observations you want included

I will update this README with:
- PDF plot image
- Mode coverage observations
- Training stability observations
- Quality of generated distribution observations

---

## Results (To Be Updated After Execution)

### PDF Plot Obtained from GAN Samples
*(Add plot image after notebook execution)*

### Observations

#### 1) Mode Coverage
*(To be filled after execution)*

#### 2) Training Stability
*(To be filled after execution)*

#### 3) Quality of Generated Distribution
*(To be filled after execution)*
