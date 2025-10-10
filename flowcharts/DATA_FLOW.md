# 📊 House Prices Prediction - Data Flow Diagram

Complete visual guide showing how data transforms from CSV files to neural network predictions.

---

## 🗂️ Starting Point: Raw CSV Files

```
📁 data/
├── train.csv      (1,460 rows × 81 columns)
│   ├── Id
│   ├── 79 features (38 numerical + 43 categorical)
│   └── SalePrice (target)
│
└── test.csv       (1,459 rows × 80 columns)
    ├── Id
    └── 79 features (no SalePrice)
```

---

## 🔄 Complete Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: Environment Setup                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Load CSV Files
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
   train_df                                    test_df
   (1460 × 81)                                (1459 × 80)
        │                                           │
        │                                           │
┌───────┴────────────────────────────────────────────────────────┐
│              PHASE 2: Exploratory Data Analysis                │
│  - Analyze SalePrice distribution (skewed: 1.88)               │
│  - Identify missing values (PoolQC: 99%, Alley: 94%)          │
│  - Find correlations (OverallQual: 0.79)                       │
│  - Detect outliers in GrLivArea                                │
│  - Analyze categorical features                                │
└────────────────────────┬───────────────────────────────────────┘
                         ↓
                  🎯 Insights Gained
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 3: Data Preprocessing                      │
└─────────────────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                 ↓
   
   📊 TODO 3.1: Create Copies
        ↓                                 ↓
   train_df_copy                    test_df_copy
   (1460 × 81)                      (1459 × 80)
        │                                 │
        │                                 │
   📊 TODO 3.2: Separate & Transform
        ↓                                 ↓
   ┌────┴────┐                       X_test
   │         │                       (1459 × 80)
   X_train   y_train                      │
   (1460×80) (1460)                       │
   │         │                            │
   │         ↓                            │
   │    LOG TRANSFORM                     │
   │    np.log1p()                        │
   │         │                            │
   │         ↓                            │
   │    y_train_log                       │
   │    (1460)                            │
   │    Skew: 1.88→0.12 ✅                │
        │                                 │
        │                                 │
   📊 TODO 3.3: Handle Missing Values
        ↓                                 ↓
   Numerical: median/0              Numerical: median/0
   Categorical: 'None'/mode         Categorical: 'None'/mode
        ↓                                 ↓
   Missing: 0 ✅                     Missing: 0 ✅
        │                                 │
        │                                 │
   📊 TODO 3.4: Feature Engineering
        ↓                                 ↓
   +5 New Features:                +5 New Features:
   - TotalSF                       - TotalSF
   - TotalBath                     - TotalBath
   - HouseAge                      - HouseAge
   - RemodAge                      - RemodAge
   - TotalPorchSF                  - TotalPorchSF
        ↓                                 ↓
   (1460 × 85)                     (1459 × 85)
        │                                 │
        │                                 │
   📊 TODO 3.5: Remove Outliers
        ↓                                 │
   Remove 2 outliers                     │
   (GrLivArea>4000 & Price<300k)         │
        ↓                                 │
   X_train: (1458 × 85)                  │
   y_train: (1458)                       │
        │                                 │
        │                                 │
   📊 TODO 3.6: One-Hot Encoding
        ↓                                 ↓
   pd.get_dummies()                pd.get_dummies()
   drop_first=True                 drop_first=True
        ↓                                 ↓
   X_train: (1458 × 264)           X_test: (1459 × 247)
   43 categorical → 221 dummies    Different columns! ⚠️
        │                                 │
        │                                 │
   📊 TODO 3.7: Align Columns
        └─────────────┬───────────────────┘
                      ↓
              X_train.align(X_test)
                      ↓
        ┌─────────────┴───────────────────┐
        ↓                                 ↓
   X_train: (1458 × 264) ✅         X_test: (1459 × 264) ✅
   y_train: (1458)                  Matching columns!
        │                                 │
        │                                 │
┌───────┴─────────────────────────────────┴───────────────────────┐
│           PHASE 4: Feature Scaling & Selection                  │
└─────────────────────────────────────────────────────────────────┘
        │                                 │
        │                                 │
   📊 TODO 4.1: Import Tools ✅
        │                                 │
        │                                 │
   📊 TODO 4.2: Train-Validation Split
        ↓                                 │
   train_test_split()                    │
   80-20, random_state=42                │
        ↓                                 │
   ┌────┴─────┐                          │
   ↓          ↓                          │
X_train_split X_val                      │
(1166 × 264)  (292 × 264)                │
y_train_split y_val                      │
(1166)        (292)                      │
   │          │                          │
   │          │                          │
   📊 TODO 4.3: StandardScaler
   │          │                          │
   ↓          ↓                          ↓
   FIT+      TRANSFORM              TRANSFORM
   TRANSFORM  ONLY                   ONLY
   ↓          ↓                          ↓
X_train_scaled X_val_scaled      X_test_scaled
(1166 × 264)   (292 × 264)       (1459 × 264)
Mean=0, Std=1  Same scaling      Same scaling
   │          │                          │
   │          │                          │
   📊 TODO 4.4: Convert to PyTorch Tensors
   │          │                          │
   ↓          ↓                          ↓
X_train_tensor X_val_tensor      test_tensor
(1166 × 264)   (292 × 264)       (1459 × 264)
float32        float32           float32
   │          │                          │
y_train_tensor y_val_tensor              │
(1166 × 1)     (292 × 1)                 │
float32        float32                   │
   │          │                          │
   │          │                          │
┌──┴──────────┴──────────────────────────┴───────────────────────┐
│              PHASE 5: Neural Network Design                     │
│  HousePricePredictor(n_features=264)                           │
│  Architecture: 264 → 256 → 128 → 64 → 1                       │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 6: Training Pipeline                     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
    Training Loop              Validation Loop
    Uses: X_train_tensor       Uses: X_val_tensor
          y_train_tensor             y_val_tensor
            │                           │
            └──────────┬────────────────┘
                       ↓
                 Track Metrics
                 (RMSE, MAE, R²)
                       ↓
                 Best Model Saved
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│            PHASE 7: Evaluation & Submission                     │
└─────────────────────────────────────────────────────────────────┘
                       ↓
               Load Best Model
                       ↓
            Predict on test_tensor
            (1459 × 264)
                       ↓
            predictions_log (1459 × 1)
                       ↓
            REVERSE LOG TRANSFORM
            np.expm1()
                       ↓
            predictions_price (1459)
            (Back to dollars!)
                       ↓
            Create submission.csv
            ┌──────────┴──────────┐
            │ Id      │ SalePrice │
            ├──────────┼───────────┤
            │ 1461    │ 127,500   │
            │ 1462    │ 156,000   │
            │ ...     │ ...       │
            └──────────┴───────────┘
                       ↓
            🎯 Submit to Kaggle!
```

---

## 📊 Data Shapes at Each Major Step

| Step | Dataset | Rows | Columns | Notes |
|------|---------|------|---------|-------|
| **Phase 1** |
| Load | train_df | 1,460 | 81 | Raw data from CSV |
| Load | test_df | 1,459 | 80 | No SalePrice column |
| **Phase 3** |
| 3.2 | X_train | 1,460 | 80 | Features only |
| 3.2 | y_train | 1,460 | 1 | Log-transformed target |
| 3.5 | X_train | 1,458 | 85 | After outlier removal + new features |
| 3.6 | X_train | 1,458 | 264 | After one-hot encoding |
| 3.7 | X_test | 1,459 | 264 | Aligned columns |
| **Phase 4** |
| 4.2 | X_train_split | 1,166 | 264 | Training subset (80%) |
| 4.2 | X_val | 292 | 264 | Validation subset (20%) |
| 4.3 | X_train_scaled | 1,166 | 264 | Standardized |
| 4.3 | X_val_scaled | 292 | 264 | Standardized |
| 4.3 | X_test_scaled | 1,459 | 264 | Standardized |
| 4.4 | X_train_tensor | 1,166 | 264 | PyTorch float32 |
| 4.4 | y_train_tensor | 1,166 | 1 | PyTorch float32 |
| **Phase 7** |
| 7.3 | submission.csv | 1,459 | 2 | Id + SalePrice |

---

## 🔑 Key Transformations Explained

### 1️⃣ **Log Transformation (Phase 3.2)**
```
Before: SalePrice = [34900, 100000, 755000, ...]
        Skewness = 1.88 (heavily right-skewed)
        
After:  y_train = [10.46, 11.51, 13.53, ...]
        Skewness = 0.12 (nearly normal!)
        
Why:    Neural networks need normal distributions
```

### 2️⃣ **One-Hot Encoding (Phase 3.6)**
```
Before: Neighborhood = ['NAmes', 'CollgCr', 'OldTown', ...]
        1 categorical column with 25 values
        
After:  Neighborhood_NAmes = [1, 0, 0, ...]
        Neighborhood_CollgCr = [0, 1, 0, ...]
        Neighborhood_OldTown = [0, 0, 1, ...]
        ...
        24 binary columns (drop_first=True)
        
Why:    Neural networks need numerical input
```

### 3️⃣ **Train-Validation Split (Phase 4.2)**
```
Before: X_train (1458 samples)
        
After:  X_train_split (1166 samples, 80%) - Train model
        X_val (292 samples, 20%) - Validate during training
        
Why:    Prevent overfitting, tune hyperparameters
```

### 4️⃣ **Standardization (Phase 4.3)**
```
Before: GrLivArea = [334, 1710, 1262, ...]
        Range: [334, 5642]
        
After:  GrLivArea_scaled = [-1.5, 0.3, -0.2, ...]
        Mean = 0, Std = 1
        Range ≈ [-3, +3]
        
Why:    Neural networks learn faster with scaled features
```

### 5️⃣ **Reverse Log Transform (Phase 7.2)**
```
Model Output: predictions_log = [11.75, 12.05, 11.92, ...]
              (Log scale)
              
After:        predictions_price = [126500, 171500, 150000, ...]
              (Dollar scale)
              
Formula:      price = exp(log_price) - 1
              np.expm1(predictions_log)
              
Why:          Kaggle expects actual prices, not log prices!
```

---

## 🎯 Three Datasets, Three Purposes

### 🟢 **Training Set** (X_train_split, y_train_split)
- **Size:** 1,166 samples
- **Purpose:** Train the neural network
- **Usage:** Backpropagation, gradient descent, weight updates
- **Has labels:** ✅ YES (y_train)

### 🟡 **Validation Set** (X_val, y_val)
- **Size:** 292 samples
- **Purpose:** Monitor overfitting during training
- **Usage:** Calculate validation metrics, early stopping
- **Has labels:** ✅ YES (y_val)

### 🔴 **Test Set** (X_test)
- **Size:** 1,459 samples
- **Purpose:** Make final predictions for Kaggle
- **Usage:** Generate submission.csv
- **Has labels:** ❌ NO (that's what we predict!)

---

## 📈 Data Size Tracking

```
CSV Files:
├─ train.csv: 1,460 samples
│  
Phase 3 (Preprocessing):
├─ After outlier removal: 1,458 samples
│  
Phase 4 (Scaling):
├─ Training subset: 1,166 samples (80%)
├─ Validation subset: 292 samples (20%)
└─ Total: 1,458 samples ✅

Test Set (Unchanged):
└─ test.csv: 1,459 samples (for Kaggle submission)
```

---

## 🔄 Complete Feature Count Evolution

```
Original Features:
├─ 38 Numerical
├─ 43 Categorical
└─ 1 Target (SalePrice)
    Total: 82 columns

After Feature Engineering (+5):
├─ 43 Numerical (38 + 5 new)
├─ 43 Categorical
└─ 1 Target
    Total: 87 columns

After One-Hot Encoding:
├─ 43 Numerical (unchanged)
├─ 221 Binary (from 43 categorical)
└─ 1 Target (separate)
    Total: 264 features for model input
```

---

## 💡 Common Confusions Explained

### ❓ "Why do we split AFTER preprocessing?"
**Answer:** We want the same preprocessing (scaling, encoding) applied consistently to all data. If we split first, the validation set might have categories the model never saw during encoding.

### ❓ "Why scale AFTER splitting?"
**Answer:** Scaling must be fit on training data only! If you scale before splitting, information from validation "leaks" into the scaler, causing overfitting.

### ❓ "Why no y_test?"
**Answer:** Kaggle keeps test labels secret. That's the competition! You predict them, submit, and Kaggle scores you.

### ❓ "What's the difference between X_test and X_val?"
- **X_val**: Validation set from YOUR training data (292 samples, has labels)
- **X_test**: Kaggle's test set (1,459 samples, NO labels)

---

## 🎓 Summary: From CSV to Neural Network

1. **Load CSVs** → Get raw data
2. **EDA** → Understand data patterns
3. **Preprocess** → Clean, engineer, encode
4. **Scale** → Normalize for neural networks
5. **Tensorize** → Convert to PyTorch format
6. **Train** → Fit neural network
7. **Predict** → Generate Kaggle submission

**Each step transforms the data to make it suitable for the next!**

---

*Last Updated: Phase 4 Complete*

