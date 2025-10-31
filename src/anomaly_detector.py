"""
AIOps Infrastructure Monitoring - Phase 2
Machine Learning Anomaly Detection Engine

Features:
- Isolation Forest for anomaly detection
- Real-time scoring
- Alert generation
- Model training and evaluation
- Feature engineering from time-series data

Requirements (add to previous):
pip install scikit-learn joblib plotly kaleido

Run after collector has been running for at least 30 minutes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from influxdb_client import InfluxDBClient
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class MLConfig:
    INFLUXDB_URL = "http://localhost:8086"
    INFLUXDB_TOKEN = "my-super-secret-token"
    INFLUXDB_ORG = "aiops"
    INFLUXDB_BUCKET = "metrics"
    
    # ML Parameters
    CONTAMINATION = 0.05  # Expected % of anomalies (5%)
    TRAINING_WINDOW = "6h"  # Use 6 hours of data for training
    PREDICTION_WINDOW = "5m"  # Analyze last 5 minutes
    
    # Feature Engineering
    ROLLING_WINDOWS = [3, 5, 10]  # Rolling average windows (in samples)
    
    # Alert Thresholds
    ANOMALY_SCORE_THRESHOLD = -0.3  # Lower = more anomalous
    CONSECUTIVE_ANOMALIES = 2  # Alert after N consecutive anomalies
    
    MODEL_PATH = "models/anomaly_detector.joblib"
    SCALER_PATH = "models/scaler.joblib"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
class FeatureEngineering:
    """Extract features from time-series metrics for ML"""
    
    @staticmethod
    def add_time_features(df):
        """Add time-based features (hour, day of week, etc.)"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        return df
    
    @staticmethod
    def add_rolling_features(df, columns, windows):
        """Add rolling statistics (mean, std, min, max)"""
        for col in columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                
                # Rate of change
                df[f'{col}_rate_change_{window}'] = df[col].diff(window).fillna(0)
        
        return df
    
    @staticmethod
    def add_lag_features(df, columns, lags=[1, 2, 3]):
        """Add lagged features (previous values)"""
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(df[col].iloc[0])
        return df
    
    @staticmethod
    def create_feature_matrix(df, metric_columns):
        """Create complete feature matrix"""
        df = FeatureEngineering.add_time_features(df)
        df = FeatureEngineering.add_rolling_features(
            df, metric_columns, MLConfig.ROLLING_WINDOWS
        )
        df = FeatureEngineering.add_lag_features(df, metric_columns)
        
        # Drop any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

# ============================================================================
# DATA LOADER
# ============================================================================
class MetricDataLoader:
    """Load and prepare data from InfluxDB for ML"""
    
    def __init__(self):
        self.client = InfluxDBClient(
            url=MLConfig.INFLUXDB_URL,
            token=MLConfig.INFLUXDB_TOKEN,
            org=MLConfig.INFLUXDB_ORG
        )
        self.query_api = self.client.query_api()
    
    def load_server_metrics(self, server_name, duration="6h"):
        """Load all metrics for a specific server"""
        metrics = {}
        
        for metric_type in ["cpu_usage", "memory_usage", "network_throughput", "disk_io"]:
            query = f'''
            from(bucket: "{MLConfig.INFLUXDB_BUCKET}")
              |> range(start: -{duration})
              |> filter(fn: (r) => r["_measurement"] == "{metric_type}")
              |> filter(fn: (r) => r["server"] == "{server_name}")
              |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
            '''
            
            result = self.query_api.query_data_frame(query, org=MLConfig.INFLUXDB_ORG)
            
            if not result.empty:
                result['_time'] = pd.to_datetime(result['_time'])
                result = result.set_index('_time')
                metrics[metric_type] = result['_value']
        
        # Combine into single DataFrame
        if metrics:
            df = pd.DataFrame(metrics)
            return df
        else:
            return None
    
    def load_all_servers(self, duration="6h"):
        """Load metrics for all servers"""
        servers = ["web-server-01", "web-server-02", "api-server-01", 
                   "db-server-01", "cache-server-01"]
        
        all_data = {}
        for server in servers:
            data = self.load_server_metrics(server, duration)
            if data is not None:
                all_data[server] = data
        
        return all_data

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================
class AnomalyDetector:
    """Machine Learning-based anomaly detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metric_columns = ["cpu_usage", "memory_usage", "network_throughput", "disk_io"]
    
    def train(self, df):
        """Train the Isolation Forest model"""
        print("Training Anomaly Detection Model...")
        print(f"Training data shape: {df.shape}")
        
        # Create features
        df_features = FeatureEngineering.create_feature_matrix(df, self.metric_columns)
        
        # Store feature columns
        self.feature_columns = [col for col in df_features.columns 
                                if col not in ['_start', '_stop', '_measurement', 
                                               '_field', 'result', 'table', 'server']]
        
        # Prepare training data
        X_train = df_features[self.feature_columns].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=MLConfig.CONTAMINATION,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled)
        
        # Evaluate on training data
        predictions = self.model.predict(X_train_scaled)
        scores = self.model.score_samples(X_train_scaled)
        
        anomalies = (predictions == -1).sum()
        print(f"\n✓ Model trained successfully!")
        print(f"  - Total samples: {len(X_train)}")
        print(f"  - Detected anomalies: {anomalies} ({anomalies/len(X_train)*100:.2f}%)")
        print(f"  - Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  - Features used: {len(self.feature_columns)}")
        
        return self
    
    def predict(self, df):
        """Predict anomalies on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create features
        df_features = FeatureEngineering.create_feature_matrix(df, self.metric_columns)
        
        # Prepare data
        X = df_features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        # Add predictions to dataframe
        df_features['anomaly'] = predictions
        df_features['anomaly_score'] = scores
        df_features['is_anomaly'] = predictions == -1
        
        return df_features
    
    def save_model(self):
        """Save trained model and scaler"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, MLConfig.MODEL_PATH)
        joblib.dump(self.scaler, MLConfig.SCALER_PATH)
        
        # Save feature columns
        with open('models/feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f)
        
        print(f"\n✓ Model saved to {MLConfig.MODEL_PATH}")
    
    def load_model(self):
        """Load trained model and scaler"""
        self.model = joblib.load(MLConfig.MODEL_PATH)
        self.scaler = joblib.load(MLConfig.SCALER_PATH)
        
        with open('models/feature_columns.json', 'r') as f:
            self.feature_columns = json.load(f)
        
        print(f"✓ Model loaded from {MLConfig.MODEL_PATH}")
        return self

# ============================================================================
# ALERT GENERATOR
# ============================================================================
class AlertGenerator:
    """Generate alerts for detected anomalies"""
    
    def __init__(self):
        self.alert_history = []
    
    def analyze_anomalies(self, df_predictions, server_name):
        """Analyze predictions and generate alerts"""
        anomalies = df_predictions[df_predictions['is_anomaly'] == True]
        
        if len(anomalies) == 0:
            return []
        
        alerts = []
        
        for idx, row in anomalies.iterrows():
            # Determine which metric is causing the anomaly
            metric_values = {
                'CPU': row.get('cpu_usage', 0),
                'Memory': row.get('memory_usage', 0),
                'Network': row.get('network_throughput', 0),
                'Disk I/O': row.get('disk_io', 0)
            }
            
            # Find the metric with highest absolute deviation
            primary_metric = max(metric_values.items(), key=lambda x: abs(x[1] - 50))
            
            alert = {
                'timestamp': idx,
                'server': server_name,
                'anomaly_score': row['anomaly_score'],
                'primary_metric': primary_metric[0],
                'metric_value': primary_metric[1],
                'severity': self._calculate_severity(row['anomaly_score']),
                'metrics': metric_values
            }
            
            alerts.append(alert)
        
        return alerts
    
    def _calculate_severity(self, score):
        """Calculate alert severity based on anomaly score"""
        if score < -0.5:
            return "CRITICAL"
        elif score < -0.3:
            return "HIGH"
        elif score < -0.1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def format_alert(self, alert):
        """Format alert for display"""
        severity_colors = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }
        
        icon = severity_colors.get(alert['severity'], "⚪")
        
        message = f"""
{icon} {alert['severity']} ANOMALY DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Server: {alert['server']}
Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Primary Issue: {alert['primary_metric']} = {alert['metric_value']:.1f}
Anomaly Score: {alert['anomaly_score']:.3f}

Current Metrics:
  CPU: {alert['metrics']['CPU']:.1f}%
  Memory: {alert['metrics']['Memory']:.1f}%
  Network: {alert['metrics']['Network']:.1f} MB/s
  Disk I/O: {alert['metrics']['Disk I/O']:.0f} ops/s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return message.strip()

# ============================================================================
# REAL-TIME MONITOR
# ============================================================================
class RealTimeMonitor:
    """Monitor infrastructure in real-time and detect anomalies"""
    
    def __init__(self):
        self.loader = MetricDataLoader()
        self.detector = AnomalyDetector()
        self.alert_gen = AlertGenerator()
    
    def train_models(self):
        """Train models on historical data"""
        print("="*70)
        print("TRAINING ANOMALY DETECTION MODELS")
        print("="*70)
        
        all_data = self.loader.load_all_servers(duration=MLConfig.TRAINING_WINDOW)
        
        if not all_data:
            print("✗ No data available. Run collector first!")
            return False
        
        # Train on combined data from all servers
        combined_data = pd.concat(all_data.values())
        self.detector.train(combined_data)
        self.detector.save_model()
        
        return True
    
    def monitor_server(self, server_name):
        """Monitor a single server"""
        # Load recent data
        df = self.loader.load_server_metrics(
            server_name, 
            duration=MLConfig.PREDICTION_WINDOW
        )
        
        if df is None or len(df) == 0:
            print(f"No data for {server_name}")
            return
        
        # Predict anomalies
        predictions = self.detector.predict(df)
        
        # Generate alerts
        alerts = self.alert_gen.analyze_anomalies(predictions, server_name)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"MONITORING: {server_name}")
        print(f"{'='*70}")
        print(f"Time window: {df.index[0]} to {df.index[-1]}")
        print(f"Samples analyzed: {len(df)}")
        print(f"Anomalies detected: {len(alerts)}")
        
        if alerts:
            print(f"\n⚠️  ALERTS GENERATED:")
            for alert in alerts:
                print(self.alert_gen.format_alert(alert))
        else:
            print("\n✓ No anomalies detected. System is healthy!")
        
        return alerts
    
    def monitor_all_servers(self):
        """Monitor all servers"""
        servers = ["web-server-01", "web-server-02", "api-server-01", 
                   "db-server-01", "cache-server-01"]
        
        all_alerts = {}
        for server in servers:
            alerts = self.monitor_server(server)
            if alerts:
                all_alerts[server] = alerts
        
        return all_alerts

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("AIOps Infrastructure Monitoring - Phase 2")
    print("Machine Learning Anomaly Detection Engine")
    print("="*70)
    print()
    
    monitor = RealTimeMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        success = monitor.train_models()
        if success:
            print("\n✓ Training complete! Now run: python anomaly_detector.py monitor")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Monitoring mode
        try:
            monitor.detector.load_model()
            print("\nStarting real-time monitoring...\n")
            
            import time
            while True:
                monitor.monitor_all_servers()
                print(f"\nNext check in 30 seconds...\n")
                time.sleep(30)
        
        except FileNotFoundError:
            print("✗ Model not found. Run training first: python anomaly_detector.py train")
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    else:
        # Demo mode - one-time analysis
        print("Running one-time analysis...")
        print("\nStep 1: Training model on historical data...")
        
        success = monitor.train_models()
        
        if success:
            print("\n\nStep 2: Analyzing current metrics...")
            monitor.monitor_all_servers()
            
            print("\n\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("1. Train model: python anomaly_detector.py train")
            print("2. Monitor continuously: python anomaly_detector.py monitor")
            print("3. Single server: python anomaly_detector.py monitor web-server-01")
        else:
            print("\n✗ Training failed. Make sure collector has been running for at least 30 minutes.")