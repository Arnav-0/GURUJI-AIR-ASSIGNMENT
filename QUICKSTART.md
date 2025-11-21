# Quick Start Guide
## mmWave Radar AI Project

This guide will get you up and running in under 10 minutes.

---

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

---

## Installation Steps

**1. Navigate to Project Directory**
```bash
cd "d:\Guruji air assignment"
```

**2. Create Virtual Environment (Recommended)**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Installation**
```bash
python test_project.py
```

You should see "ALL TESTS PASSED!" if everything is installed correctly.

---

## Running the Notebooks

### Part 1: Radar Simulation
```bash
jupyter notebook notebooks/01_radar_simulation.ipynb
```

**What it does:**
- Generates synthetic radar signals
- Applies FFT transformations
- Creates range-Doppler heatmaps
- Visualizes empty room, metal objects, and clutter scenarios

**Expected output:** 10 visualization figures saved to `outputs/figures/`

**Execution time:** 10-15 minutes

---

### Part 2: Classification Model
```bash
jupyter notebook notebooks/02_classification_model.ipynb
```

**What it does:**
- Generates 400+ training samples
- Trains CNN and SVM models
- Evaluates performance with confusion matrix
- Saves trained models

**Expected output:** Trained models in `data/models/`, accuracy >95%

**Execution time:** 30-60 minutes (depending on hardware)

---

### Part 3: Hidden Object Detection
```bash
jupyter notebook notebooks/03_hidden_object_detection.ipynb
```

**What it does:**
- Tests model on cluttered scenarios
- Applies background subtraction
- Implements noise filtering
- Compares preprocessing techniques

**Expected output:** Performance improvement analysis, accuracy >92% on cluttered data

**Execution time:** 15-20 minutes

---

## Running the Demo

```bash
cd src
python demo.py
```

This will run all 5 demo scenarios, generate visualizations, and save demo figures for video creation.

---

## Generate Deployment Document

```bash
cd docs
python create_deployment_pdf.py
```

This creates `deployment_design.pdf` with complete architecture documentation.

---

## Project Structure Overview

```
mmwave-radar-ai/
├── notebooks/          # Jupyter notebooks for each part
├── src/               # Python modules and utilities
├── data/              # Generated datasets and models
├── outputs/           # Results and visualizations
├── docs/              # Documentation and PDFs
└── requirements.txt   # Dependencies
```

---

## Troubleshooting

### Issue: TensorFlow not found
```bash
pip install tensorflow --upgrade
```

### Issue: Jupyter kernel not found
```bash
python -m ipykernel install --user --name=.venv
```

### Issue: Out of memory during training
- Reduce batch size in notebook (change `batch_size=32` to `batch_size=16`)
- Close other applications
- Use CPU-only mode if GPU memory is insufficient

### Issue: Slow execution
- Enable GPU acceleration if available
- Reduce number of training samples for testing
- Use pre-trained models from `data/models/`

---

## Expected Results

| Part | Metric | Expected Value |
|------|--------|----------------|
| Part 1 | Figures generated | 10 |
| Part 2 | CNN Accuracy | >95% |
| Part 2 | SVM Accuracy | >90% |
| Part 3 | Cluttered Accuracy | >92% |
| Part 4 | PDF pages | 15+ |

---

## Next Steps

1. Complete all notebook executions
2. Review generated visualizations
3. Generate deployment PDF
4. Run demo script
5. Create demo video (compile demo figures)
6. Push to GitHub repository

---

## Assignment Completion Checklist

- [ ] Part 1: Radar simulation complete (10 figures)
- [ ] Part 2: Models trained and saved (>95% accuracy)
- [ ] Part 3: Hidden object detection tested (>92% accuracy)
- [ ] Part 4: deployment_design.pdf created
- [ ] Part 5: Demo figures ready for video
- [ ] All code documented and clean
- [ ] GitHub repository prepared
- [ ] README.md comprehensive

---

**Total Estimated Time:** 2-3 hours (including training)

**Difficulty Level:** Advanced

For detailed instructions, see EXECUTION_GUIDE.md
