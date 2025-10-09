# 📥 Kaggle Dataset Setup Guide

This guide will help you download the House Prices dataset from Kaggle.

---

## Method 1: Web Browser (Easiest) 🌐

1. **Visit the competition page:**
   - Go to: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

2. **Login or create account:**
   - If you don't have a Kaggle account, create one (it's free!)
   - Login with your credentials

3. **Accept competition rules:**
   - Click "I Understand and Accept" to join the competition
   - This is required to download the data

4. **Download the files:**
   - Download `train.csv` (training data with SalePrice)
   - Download `test.csv` (test data for predictions)
   - Download `data_description.txt` (optional but helpful - describes all features)
   - Download `sample_submission.csv` (shows submission format)

5. **Move files to project:**
   - Place all downloaded files in the `data/` folder of this project
   - Your structure should look like:
     ```
     house_prices_prediction/
     ├── data/
     │   ├── train.csv ✅
     │   ├── test.csv ✅
     │   ├── data_description.txt
     │   └── sample_submission.csv
     ```

---

## Method 2: Kaggle CLI (Advanced) 💻

If you prefer using the command line:

### 1. Install Kaggle CLI
```bash
pip install kaggle
```

### 2. Setup API Credentials

1. Go to your Kaggle account settings: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json` file
5. Move it to the correct location:

   **Mac/Linux:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Windows:**
   ```cmd
   mkdir %USERPROFILE%\.kaggle
   move Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

### 3. Download Dataset
```bash
cd house_prices_prediction

# Download and extract
kaggle competitions download -c house-prices-advanced-regression-techniques

# Extract to data folder
unzip house-prices-advanced-regression-techniques.zip -d data/

# Clean up zip file
rm house-prices-advanced-regression-techniques.zip
```

### 4. Verify Download
```bash
ls -lh data/
```

You should see:
- `train.csv` (~460 KB)
- `test.csv` (~450 KB)
- `data_description.txt`
- `sample_submission.csv`

---

## ✅ Verification Checklist

Before starting the notebook, make sure you have:

- [ ] `data/train.csv` exists
- [ ] `data/test.csv` exists
- [ ] Joined the Kaggle competition (accepted rules)
- [ ] (Optional) Read `data_description.txt` to understand features

---

## 📊 Dataset Quick Facts

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train.csv` | 1,460 | 81 | Training data with SalePrice |
| `test.csv` | 1,459 | 80 | Test data (predict SalePrice) |

**Target Variable:** `SalePrice` - the property's sale price in dollars

---

## 🚀 Next Steps

Once you have the data downloaded:

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/house_prices.ipynb
   ```

2. Start with Phase 1: Environment Setup

3. Work through each TODO block

4. Ask for help when you get stuck!

---

## 🆘 Troubleshooting

**Problem:** "403 Forbidden" when downloading
- **Solution:** Make sure you've accepted the competition rules on Kaggle

**Problem:** Kaggle CLI says "Could not find kaggle.json"
- **Solution:** Make sure `kaggle.json` is in `~/.kaggle/` (Mac/Linux) or `%USERPROFILE%\.kaggle\` (Windows)

**Problem:** "Competition not found"
- **Solution:** Double-check the competition name: `house-prices-advanced-regression-techniques`

---

Happy Learning! 🎓

