# Execution Guide
## mmWave Radar AI Project

Complete step-by-step guide for executing all parts of the assignment.

---

## Project Status

All components tested and verified. Follow this guide to complete the assignment.

---

## Quick Execution (3 Hours Total)

### Phase 1: Setup (10 minutes)

```bash
# Navigate to project
cd "d:\Guruji air assignment"

# Activate virtual environment (if not active)
.\.venv\Scripts\activate

# Verify installation (should show "ALL TESTS PASSED!")
python test_project.py
```

---

### Phase 2: Execute Notebooks (2 hours)

#### Notebook 1: Radar Simulation (30 minutes)
```bash
jupyter notebook notebooks/01_radar_simulation.ipynb
```

**Actions:**
1. Run all cells sequentially (Shift + Enter)
2. Wait for visualizations to generate
3. Verify 10 figures saved in `outputs/figures/`

**Expected Output:**
- Range profiles (empty room, metal object)
- Doppler profiles
- Range-Doppler heatmaps
- Comparison visualizations
- CFAR detection results

---

#### Notebook 2: Classification Model (60 minutes)
```bash
jupyter notebook notebooks/02_classification_model.ipynb
```

**Actions:**
1. Run all cells sequentially
2. Wait for dataset generation (400 samples)
3. Wait for CNN training (~10-15 min with GPU, 30-45 min CPU)
4. Review accuracy metrics (should be >95%)

**Expected Output:**
- Training dataset: 400 samples
- CNN accuracy: >95%
- SVM accuracy: >90%
- Models saved in `data/models/`
- Confusion matrices and ROC curves

---

#### Notebook 3: Hidden Object Detection (30 minutes)
```bash
jupyter notebook notebooks/03_hidden_object_detection.ipynb
```

**Actions:**
1. Run all cells sequentially
2. Review preprocessing techniques
3. Compare performance improvements
4. Verify accuracy >92% on cluttered data

**Expected Output:**
- Cluttered scenarios tested
- Background subtraction demonstrated
- Noise filtering applied
- Performance comparison charts

---

### Phase 3: Generate Demo (20 minutes)

```bash
# Generate demo visualizations
python src/demo.py
```

**Expected Output:**
- 5 demo figures in `outputs/figures/`
- Live simulation results
- Performance metrics

---

### Phase 4: Deployment PDF (Already Complete)

The deployment design PDF has been pre-generated:
- Location: `docs/deployment_design.pdf`
- Pages: 15
- Content: System architecture, hardware requirements, real-time pipeline

---

### Phase 5: Demo Video (10 minutes recording)

1. Open `docs/VIDEO_GUIDE.md` for script
2. Use screen recording software (OBS, Camtasia, or Windows Game Bar)
3. Record walkthrough following the script
4. Upload to YouTube or Google Drive
5. Include link in submission

**Video Structure:**
- Introduction (30 sec)
- Part 1 Demo (90 sec)
- Part 2 Demo (90 sec)
- Part 3 Demo (60 sec)
- Deployment Overview (60 sec)
- Conclusion (30 sec)

Total: 5-6 minutes

---

## Detailed Notebook Instructions

### Notebook 1: Radar Simulation

**Cell-by-cell breakdown:**

1. **Imports and Setup** (5 sec)
   - Load required libraries
   - Initialize parameters

2. **RadarSimulator Initialization** (5 sec)
   - Create simulator with 77 GHz, 4 GHz bandwidth
   - Display radar parameters

3. **Empty Room Scenario** (30 sec)
   - Generate signals
   - Apply FFT
   - Plot range profile
   - Save figure 01

4. **Metal Object Detection** (30 sec)
   - Generate signals with metal target
   - Apply FFT
   - Plot range and Doppler profiles
   - Save figures 02-03

5. **Range-Doppler Heatmaps** (2 min)
   - Generate heatmaps for 3 scenarios
   - Empty, metal, clutter
   - Save figures 04-06

6. **Comparison Visualizations** (2 min)
   - Compare multiple scenarios
   - Test window functions
   - Save figures 07-08

7. **Multi-Target and CFAR** (1 min)
   - Multiple targets simultaneously
   - CFAR detection algorithm
   - Save figures 09-10

**Total execution time:** 5-7 minutes (depending on hardware)

---

### Notebook 2: Classification Model

**Cell-by-cell breakdown:**

1. **Imports and Setup** (10 sec)
   - Load libraries
   - Set random seeds for reproducibility

2. **Dataset Generation** (5-10 min)
   - Generate 200 metal samples
   - Generate 200 non-metal samples
   - Split into train/validation (80/20)
   - Display sample heatmaps

3. **CNN Model Creation** (5 sec)
   - Build 4-block CNN architecture
   - 2.27M parameters
   - Display model summary

4. **Model Training** (10-45 min)
   - Train for up to 50 epochs
   - Early stopping enabled
   - Learning rate reduction
   - Model checkpointing

5. **CNN Evaluation** (30 sec)
   - Predict on validation set
   - Calculate accuracy, precision, recall
   - Generate confusion matrix
   - Plot ROC curve

6. **SVM Training** (2-5 min)
   - Train SVM with RBF kernel
   - Evaluate performance
   - Compare with CNN

7. **Results Visualization** (1 min)
   - Training history plots
   - Confusion matrices
   - ROC curves
   - Model comparison charts

**Total execution time:** 20-60 minutes (GPU: 20-25 min, CPU: 45-60 min)

---

### Notebook 3: Hidden Object Detection

**Cell-by-cell breakdown:**

1. **Imports and Setup** (5 sec)
   - Load libraries and trained model

2. **Load Pre-trained Model** (5 sec)
   - Load best CNN from Part 2
   - Verify weights loaded correctly

3. **Generate Cluttered Scenarios** (2 min)
   - Create complex scenes with clutter
   - Multiple objects, noise, interference

4. **Baseline Performance** (1 min)
   - Test model without preprocessing
   - Record baseline accuracy

5. **Background Subtraction** (2 min)
   - Apply background removal
   - Test improvement
   - Visualize results

6. **Noise Filtering** (3 min)
   - Apply median filter
   - Apply Gaussian filter
   - Compare effectiveness

7. **Combined Preprocessing** (2 min)
   - Use all techniques together
   - Measure final performance
   - Should achieve >92% accuracy

8. **Results Comparison** (1 min)
   - Plot performance comparisons
   - Visualize preprocessing effects
   - Save final figures

**Total execution time:** 12-15 minutes

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: Jupyter won't start**
```bash
pip install --upgrade jupyter notebook
python -m notebook
```

**Issue 2: Kernel crashes during training**
- Reduce batch size to 16
- Close unnecessary applications
- Monitor RAM usage

**Issue 3: CUDA out of memory**
```python
# Add to notebook beginning
import tensorflow as tf
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
)
```

**Issue 4: Import errors in notebooks**
```python
# Add to first cell
import sys
sys.path.append('../src')
```

**Issue 5: Slow training**
- Verify GPU is being used:
```python
print(tf.config.list_physical_devices('GPU'))
```
- If empty, TensorFlow is using CPU

---

## Verification Checklist

After completing all phases:

### Files Generated
- [ ] 10+ figures in `outputs/figures/`
- [ ] Trained models in `data/models/cnn_classifier.h5`
- [ ] Results text file in `outputs/results/`
- [ ] Demo video (MP4 or link)

### Performance Metrics
- [ ] Part 1: All visualizations clear and labeled
- [ ] Part 2: CNN accuracy >95%
- [ ] Part 2: SVM accuracy >90%
- [ ] Part 3: Cluttered accuracy >92%

### Documentation
- [ ] All notebooks executed (outputs visible)
- [ ] README.md complete
- [ ] deployment_design.pdf present
- [ ] Video demonstrates all parts

---

## Submission Preparation

### 1. Clean up project
```bash
# Remove unnecessary files
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force .ipynb_checkpoints
Remove-Item -Recurse -Force src/__pycache__
```

### 2. Test one more time
```bash
python test_project.py
```

### 3. Create GitHub repository
```bash
git init
git add .
git commit -m "Complete mmWave Radar AI Project"
git branch -M main
git remote add origin https://github.com/yourusername/mmwave-radar-ai.git
git push -u origin main
```

### 4. Prepare submission package

Create a zip file with:
- All code and notebooks
- Generated visualizations
- deployment_design.pdf
- Link to GitHub repo
- Link to demo video

---

## Final Quality Check

Before submission, verify:

1. **Code Quality**
   - No hardcoded paths
   - Comments are clear
   - No debug print statements

2. **Results**
   - All figures have titles and labels
   - Accuracy metrics match expectations
   - Models saved correctly

3. **Documentation**
   - README is comprehensive
   - Quick start works
   - All links functional

4. **Professionalism**
   - Consistent formatting
   - No typos
   - Clean file structure

---

## Expected Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Setup | 10 min | 10 min |
| Notebook 1 | 30 min | 40 min |
| Notebook 2 | 60 min | 100 min |
| Notebook 3 | 30 min | 130 min |
| Demo | 20 min | 150 min |
| Video | 10 min | 160 min |
| Cleanup | 10 min | 170 min |

**Total: ~3 hours**

---

## Success Criteria

Assignment is complete when:
- All 5 parts executed successfully
- Performance metrics meet requirements
- Documentation is comprehensive
- Video demonstrates understanding
- Code is clean and professional

---

Good luck with your assignment!

For quick reference, see QUICKSTART.md
