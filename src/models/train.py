import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import yaml
import click
from mlflow.tracking import MlflowClient

class FraudDetectionModel:
    def __init__(self, config_path="config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_data(self, data_path="data/raw/creditcard.csv"):
        """Load and preprocess the credit card fraud dataset"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        # Scale the 'Amount' and 'Time' features
        X['Amount'] = self.scaler.fit_transform(X[['Amount']])
        X['Time'] = self.scaler.fit_transform(X[['Time']])
        
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        
        # Random Forest
        rf_params = self.config['models']['random_forest']
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        
        # XGBoost  
        xgb_params = self.config['models']['xgboost']
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
        
        models = {
            'random_forest': rf_model,
            'xgboost': xgb_model
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
        return models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate trained models"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"AUC Score: {auc_score:.4f}")
            
        return results

@click.command()
@click.option('--data-path', default='data/raw/creditcard.csv')
@click.option('--experiment-name', default='credit-card-fraud-detection')
def main(data_path, experiment_name):
    """Main training pipeline"""

    # Set tracking URI to your MLflow server
    mlflow.set_tracking_uri("http://98.80.224.211:5000")
    
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Initialize model trainer
        trainer = FraudDetectionModel()
        
        # Load and split data
        X, y = trainer.load_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Log data info
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_param("fraud_rate", y.mean())
        
        # Train models
        models = trainer.train_models(X_train, y_train)
        
        # Evaluate models
        results = trainer.evaluate_models(models, X_test, y_test)
        
        # Find and register best model
        best_model_name = max(results.items(), key=lambda x: x[1]['auc_score'])[0]
        best_model = models[best_model_name]
        best_auc = results[best_model_name]['auc_score']
        
        # Log and register best model
        model_info = None
        if best_model_name == 'xgboost':
            model_info = mlflow.xgboost.log_model(
                best_model,
                artifact_path="model",
            registered_model_name="credit-card-fraud-model"  # ← This creates/updates registry
            )
        else:
            model_info = mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
            registered_model_name="credit-card-fraud-model"
            )
            
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_auc", best_auc)
        
        # Save scaler
        scaler_path = "models/scaler.joblib"
        joblib.dump(trainer.scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")
        
        
        # AUTO-PROMOTION TO STAGING IF METRIC THRESHOLD IS MET
        client = MlflowClient()
        AUC_THRESHOLD = 0.96

        # Get the latest version of the registered model
        latest_version = client.get_latest_versions("credit-card-fraud-model", stages=["None"])[0]
        if best_auc >= AUC_THRESHOLD:
            print(f"AUC {best_auc:.4f} >= {AUC_THRESHOLD} → promoting version {latest_version.version} to 'Staging'")
            client.transition_model_version_stage(
                name="credit-card-fraud-model",
                version=latest_version.version,
                stage="Staging"
            )
        else:
            print(f"AUC {best_auc:.4f} < {AUC_THRESHOLD} → not promoting to Staging")

        # Optional:add a reason
        client.update_model_version(
            name="credit-card-fraud-model",
            version=latest_version.version,
            description=f"Model trained with AUC={best_auc:.4f}. Promoted to Staging: {best_auc >= AUC_THRESHOLD}"
        )

        # Export model to local path for API
        print("Exporting model for API...")
        os.makedirs("models/artifacts", exist_ok=True)

        model_uri = f"runs:/{run.info.run_id}/model"

        # Download model files from MLflow run
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="models/artifacts/")

        print(f"Model exported to {local_model_path}")

if __name__ == "__main__":
    main()
