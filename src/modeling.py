import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_modeling_data(df):
    """Prepare data for modeling tasks with robust column creation and debugging"""
    logger.info("Starting prepare_modeling_data")
    if df.empty:
        logger.error("Input DataFrame is empty")
        raise ValueError("Input DataFrame is empty")
    
    df = df.copy()
    
    # Verify required columns
    required_cols = ['TransactionMonth', 'RegistrationYear', 'TotalPremium', 
                     'Province', 'VehicleType', 'Gender']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Input DataFrame columns: {list(df.columns)}")
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Data types:\n{df[required_cols].dtypes}")
    logger.info(f"NaN counts:\n{df[required_cols].isna().sum()}")
    logger.info(f"TransactionMonth sample:\n{df['TransactionMonth'].head().to_string()}")
    
    # Extract TransactionYear from TransactionMonth
    try:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        df['TransactionYear'] = df['TransactionMonth'].dt.year
        logger.info(f"Extracted TransactionYear from TransactionMonth. NaN count: {df['TransactionYear'].isna().sum()}")
    except Exception as e:
        logger.error(f"Error extracting TransactionYear: {e}")
        raise ValueError(f"Error extracting TransactionYear: {e}")
    
    # Convert to numeric types
    try:
        for col in ['TransactionYear', 'RegistrationYear', 'TotalPremium']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Converted {col} to numeric. NaN count: {df[col].isna().sum()}")
    except Exception as e:
        logger.error(f"Error converting columns to numeric: {e}")
        raise ValueError(f"Error converting columns to numeric: {e}")
    
    # Feature engineering
    try:
        # VehicleAge
        df['VehicleAge'] = df['TransactionYear'] - df['RegistrationYear']
        logger.info(f"Created VehicleAge. NaN count: {df['VehicleAge'].isna().sum()}")
        logger.info(f"VehicleAge sample:\n{df['VehicleAge'].head().to_string()}")
        
        # UnderwrittenCovers
        if 'UnderwrittenCoverID' in df.columns:
            df['UnderwrittenCovers'] = df.groupby('PolicyID')['UnderwrittenCoverID'].transform('nunique')
            logger.info(f"Derived UnderwrittenCovers from UnderwrittenCoverID")
        else:
            df['UnderwrittenCovers'] = 1
            logger.info(f"Set UnderwrittenCovers to default value 1")
        df['UnderwrittenCovers'] = df['UnderwrittenCovers'].fillna(1)
        logger.info(f"Filled UnderwrittenCovers NaNs. NaN count: {df['UnderwrittenCovers'].isna().sum()}")
        
        # PremiumPerCover
        df['PremiumPerCover'] = np.where(
            df['UnderwrittenCovers'] != 0,
            df['TotalPremium'] / df['UnderwrittenCovers'],
            0
        )
        logger.info(f"Created PremiumPerCover. NaN count: {df['PremiumPerCover'].isna().sum()}")
        logger.info(f"PremiumPerCover sample:\n{df['PremiumPerCover'].head().to_string()}")
        
        # Handle categorical columns
        df['Gender'] = df['Gender'].fillna('U')
        df['VehicleType'] = df['VehicleType'].fillna('Passenger Vehicle')
        logger.info(f"Filled Gender and VehicleType NaNs")
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise ValueError(f"Error during feature engineering: {e}")
    
    # Verify engineered features
    engineered_features = ['VehicleAge', 'UnderwrittenCovers', 'PremiumPerCover']
    missing_engineered = [col for col in engineered_features if col not in df.columns]
    if missing_engineered:
        logger.error(f"Engineered features missing: {missing_engineered}")
        raise ValueError(f"Engineered features missing: {missing_engineered}")
    
    logger.info(f"Output DataFrame columns: {list(df.columns)}")
    logger.info(f"Output DataFrame shape: {df.shape}")
    return df

def build_preprocessor(numeric_features, categorical_features):
    """Build data preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

def train_severity_model(X, y, model_type='xgb'):
    """Train claim severity prediction model"""
    numeric_features = [col for col in X.columns if col in ['VehicleAge', 'UnderwrittenCovers', 'PremiumPerCover']]
    categorical_features = [col for col in X.columns if col in ['Province', 'VehicleType', 'Gender']]
    
    logger.info(f"Training with numeric features: {numeric_features}")
    logger.info(f"Training with categorical features: {categorical_features}")
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    else:  # xgb
        model = XGBRegressor(random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)])
    
    pipeline.fit(X, y)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate regression model performance"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'R2': r2}

def analyze_feature_importance(model, X, feature_names=None):
    """Analyze feature importance using SHAP values"""
    preprocessor = model.named_steps['preprocessor']
    X_processed = preprocessor.transform(X)
    
    if feature_names is None:
        numeric_features = [col for col in X.columns if col in ['VehicleAge', 'UnderwrittenCovers', 'PremiumPerCover']]
        categorical_features = [col for col in X.columns if col in ['Province', 'VehicleType', 'Gender']]
        ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        categorical_names = ohe.get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(categorical_names)
    
    regressor = model.named_steps['regressor']
    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_processed)
    
    return shap_values, feature_names

def save_model(model, filename):
    """Save trained model to file"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}.joblib')