"""
AIOps Infrastructure Monitoring - Phase 2B
LSTM-based Predictive Scaling Engine

Features:
- Time-series forecasting with LSTM
- 30-minute ahead predictions
- Auto-scaling recommendations
- Model training and evaluation
- Multi-step forecasting

Requirements (add to previous):
pip install tensorflow keras

Note: This will work better with GPU but runs fine on CPU
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class LSTMConfig:
    INFLUXDB_URL = "http://localhost:8086"
    INFLUXDB_TOKEN = "my-super-secret-token"
    INFLUXDB_ORG = "aiops"
    INFLUXDB_BUCKET = "metrics"
    
    # LSTM Parameters
    LOOKBACK_WINDOW = 180  # Use 30 minutes of history (180 samples at 10s)
    FORECAST_HORIZON = 180  # Predict 30 minutes ahead
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Model Architecture
    LSTM_UNITS = [128, 64]
    DROPOUT_RATE = 0.2
    
    # Thresholds
    HIGH_USAGE_THRESHOLD = 80  # % - trigger scaling alert
    PREDICTION_CONFIDENCE = 0.85  # Confidence level
    
    MODEL_DIR = "models/lstm"

# ============================================================================
# TIME SERIES PREPROCESSOR
# ============================================================================
class TimeSeriesPreprocessor:
    """Prepare time-series data for LSTM"""
    
    def __init__(self, lookback=180, forecast_horizon=180):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
    
    def prepare_data(self, data):
        """
        Convert time-series to supervised learning format
        X: [samples, lookback, features]
        y: [samples, forecast_horizon, features]
        """
        # Scale data
        if not self.fitted:
            scaled_data = self.scaler.fit_transform(data)
            self.fitted = True
        else:
            scaled_data = self.scaler.transform(data)
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(scaled_data) - self.lookback - self.forecast_horizon):
            X.append(scaled_data[i:i + self.lookback])
            y.append(scaled_data[i + self.lookback:i + self.lookback + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data):
        """Convert scaled predictions back to original scale"""
        return self.scaler.inverse_transform(data)

# ============================================================================
# LSTM MODEL BUILDER
# ============================================================================
class LSTMPredictor:
    """LSTM model for time-series prediction"""
    
    def __init__(self, n_features=1):
        self.n_features = n_features
        self.model = None
        self.preprocessor = TimeSeriesPreprocessor(
            lookback=LSTMConfig.LOOKBACK_WINDOW,
            forecast_horizon=LSTMConfig.FORECAST_HORIZON
        )
        self.history = None
    
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                LSTMConfig.LSTM_UNITS[0],
                return_sequences=True,
                input_shape=(LSTMConfig.LOOKBACK_WINDOW, self.n_features)
            ),
            layers.Dropout(LSTMConfig.DROPOUT_RATE),
            
            # Second LSTM layer
            layers.LSTM(
                LSTMConfig.LSTM_UNITS[1],
                return_sequences=True
            ),
            layers.Dropout(LSTMConfig.DROPOUT_RATE),
            
            # Output layer - predict next N steps
            layers.TimeDistributed(
                layers.Dense(self.n_features)
            )
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, data, validation_split=0.2):
        """Train LSTM model"""
        print(f"\nPreparing training data...")
        print(f"  Input shape: {data.shape}")
        
        # Prepare sequences
        X, y = self.preprocessor.prepare_data(data)
        
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        # Build model
        if self.model is None:
            self.build_model()
        
        print(f"\nModel Architecture:")
        self.model.summary()
        
        # Train
        print(f"\nTraining model...")
        self.history = self.model.fit(
            X, y,
            batch_size=LSTMConfig.BATCH_SIZE,
            epochs=LSTMConfig.EPOCHS,
            validation_split=validation_split,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        )
        
        # Evaluate
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]
        
        print(f"\n✓ Training complete!")
        print(f"  Final training loss: {train_loss:.4f}")
        print(f"  Final validation loss: {val_loss:.4f}")
        
        return self
    
    def predict(self, recent_data):
        """Predict future values"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale data
        scaled_data = self.preprocessor.scaler.transform(recent_data)
        
        # Ensure we have enough data
        if len(scaled_data) < LSTMConfig.LOOKBACK_WINDOW:
            raise ValueError(f"Need at least {LSTMConfig.LOOKBACK_WINDOW} samples")
        
        # Take last lookback window
        X = scaled_data[-LSTMConfig.LOOKBACK_WINDOW:].reshape(
            1, LSTMConfig.LOOKBACK_WINDOW, self.n_features
        )
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Reshape and inverse transform
        predictions = predictions.reshape(-1, self.n_features)
        predictions = self.preprocessor.inverse_transform(predictions)
        
        return predictions
    
    def save_model(self, path=None):
        """Save model and preprocessor"""
        import os
        
        if path is None:
            path = LSTMConfig.MODEL_DIR
        
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        self.model.save(f"{path}/lstm_model.h5")
        
        # Save preprocessor
        joblib.dump(self.preprocessor, f"{path}/preprocessor.joblib")
        
        # Save metadata
        metadata = {
            'n_features': self.n_features,
            'lookback': LSTMConfig.LOOKBACK_WINDOW,
            'forecast_horizon': LSTMConfig.FORECAST_HORIZON
        }
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"\n✓ Model saved to {path}")
    
    def load_model(self, path=None):
        """Load model and preprocessor"""
        if path is None:
            path = LSTMConfig.MODEL_DIR
        
        # Load Keras model
        self.model = keras.models.load_model(f"{path}/lstm_model.h5")
        
        # Load preprocessor
        self.preprocessor = joblib.load(f"{path}/preprocessor.joblib")
        
        # Load metadata
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.n_features = metadata['n_features']
        
        print(f"✓ Model loaded from {path}")
        return self

# ============================================================================
# PREDICTIVE SCALER
# ============================================================================
class PredictiveScaler:
    """Generate scaling recommendations based on predictions"""
    
    def __init__(self):
        self.loader = self._init_loader()
        self.predictor = None
    
    def _init_loader(self):
        """Initialize data loader"""
        from anomaly_detector import MetricDataLoader
        return MetricDataLoader()
    
    def train_predictor(self, server_name, metric='cpu_usage'):
        """Train LSTM predictor for a specific metric"""
        print(f"\n{'='*70}")
        print(f"TRAINING LSTM PREDICTOR")
        print(f"Server: {server_name} | Metric: {metric}")
        print(f"{'='*70}")
        
        # Load historical data (need at least 6 hours)
        df = self.loader.load_server_metrics(server_name, duration="24h")
        
        if df is None or len(df) < 1000:
            print("✗ Insufficient data. Need at least 24 hours of history.")
            return False
        
        # Extract metric
        data = df[[metric]].values
        
        print(f"Loaded {len(data)} samples ({len(data)*10/3600:.1f} hours)")
        
        # Train model
        self.predictor = LSTMPredictor(n_features=1)
        self.predictor.train(data)
        
        # Save model
        self.predictor.save_model(f"models/lstm/{server_name}_{metric}")
        
        return True
    
    def predict_and_recommend(self, server_name, metric='cpu_usage'):
        """Predict future usage and generate scaling recommendations"""
        
        # Load model
        if self.predictor is None:
            self.predictor = LSTMPredictor(n_features=1)
            try:
                self.predictor.load_model(f"models/lstm/{server_name}_{metric}")
            except:
                print(f"✗ No trained model found for {server_name} - {metric}")
                print("  Run training first!")
                return None
        
        # Load recent data
        df = self.loader.load_server_metrics(server_name, duration="1h")
        
        if df is None or len(df) < LSTMConfig.LOOKBACK_WINDOW:
            print("✗ Insufficient recent data")
            return None
        
        # Get current and historical values
        recent_data = df[[metric]].values
        current_value = recent_data[-1][0]
        
        # Predict future
        predictions = self.predictor.predict(recent_data)
        
        # Analyze predictions
        predicted_values = predictions.flatten()
        max_predicted = predicted_values.max()
        time_to_peak = np.argmax(predicted_values) * 10 / 60  # minutes
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            current_value,
            max_predicted,
            time_to_peak,
            metric
        )
        
        return {
            'server': server_name,
            'metric': metric,
            'current_value': current_value,
            'predictions': predicted_values,
            'max_predicted': max_predicted,
            'time_to_peak': time_to_peak,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, current, predicted_max, time_to_peak, metric):
        """Generate scaling recommendation"""
        
        if predicted_max < LSTMConfig.HIGH_USAGE_THRESHOLD:
            return {
                'action': 'NONE',
                'severity': 'INFO',
                'message': f'No action needed. Peak {metric} predicted at {predicted_max:.1f}%',
                'confidence': 'HIGH'
            }
        
        # Calculate severity
        if predicted_max > 95:
            severity = 'CRITICAL'
            action = 'SCALE_IMMEDIATELY'
        elif predicted_max > 90:
            severity = 'HIGH'
            action = 'SCALE_SOON'
        elif predicted_max > 85:
            severity = 'MEDIUM'
            action = 'PREPARE_SCALING'
        else:
            severity = 'LOW'
            action = 'MONITOR'
        
        # Calculate recommended capacity
        current_capacity = 100
        if predicted_max > 80:
            recommended_capacity = int((predicted_max / 70) * current_capacity)
            scale_up_percent = ((recommended_capacity - current_capacity) / current_capacity) * 100
        else:
            recommended_capacity = current_capacity
            scale_up_percent = 0
        
        message = f"""
⚠️  PREDICTIVE SCALING ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric: {metric}
Current: {current:.1f}%
Predicted Peak: {predicted_max:.1f}%
Time to Peak: {time_to_peak:.1f} minutes

Recommendation: {action}
Scale Up: {scale_up_percent:.0f}% (to {recommended_capacity}% capacity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        return {
            'action': action,
            'severity': severity,
            'message': message.strip(),
            'scale_up_percent': scale_up_percent,
            'recommended_capacity': recommended_capacity,
            'time_to_peak': time_to_peak,
            'confidence': 'HIGH' if predicted_max > 90 else 'MEDIUM'
        }
    
    def visualize_prediction(self, result, save_path=None):
        """Visualize predictions vs current state"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Time axis (in minutes)
        forecast_time = np.arange(0, len(result['predictions'])) * 10 / 60  # Convert to minutes
        
        # Plot predictions
        ax.plot(forecast_time, result['predictions'], 
                label='Predicted', color='blue', linewidth=2)
        
        # Current value
        ax.axhline(y=result['current_value'], 
                   color='green', linestyle='--', 
                   label=f"Current: {result['current_value']:.1f}%")
        
        # Threshold line
        ax.axhline(y=LSTMConfig.HIGH_USAGE_THRESHOLD, 
                   color='red', linestyle='--', 
                   label=f"Threshold: {LSTMConfig.HIGH_USAGE_THRESHOLD}%")
        
        # Highlight peak
        peak_idx = np.argmax(result['predictions'])
        ax.scatter([forecast_time[peak_idx]], [result['max_predicted']], 
                   color='red', s=100, zorder=5,
                   label=f"Peak: {result['max_predicted']:.1f}%")
        
        ax.set_xlabel('Time (minutes ahead)', fontsize=12)
        ax.set_ylabel(f"{result['metric']} (%)", fontsize=12)
        ax.set_title(f"Predictive Scaling - {result['server']}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# ============================================================================
# CONTINUOUS PREDICTOR
# ============================================================================
class ContinuousPredictor:
    """Run predictions continuously and generate alerts"""
    
    def __init__(self):
        self.scaler = PredictiveScaler()
        self.alert_history = []
    
    def monitor_server(self, server_name, metrics=['cpu_usage', 'memory_usage']):
        """Monitor and predict for a server"""
        print(f"\n{'='*70}")
        print(f"PREDICTIVE MONITORING: {server_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        all_alerts = []
        
        for metric in metrics:
            result = self.scaler.predict_and_recommend(server_name, metric)
            
            if result is None:
                continue
            
            # Display current status
            print(f"\n📊 {metric.upper().replace('_', ' ')}")
            print(f"   Current: {result['current_value']:.1f}%")
            print(f"   Predicted Peak: {result['max_predicted']:.1f}% "
                  f"(in {result['time_to_peak']:.1f} min)")
            
            # Check if action needed
            rec = result['recommendation']
            if rec['action'] != 'NONE':
                print(f"\n{rec['message']}")
                all_alerts.append(result)
                
                # Save visualization
                self.scaler.visualize_prediction(
                    result, 
                    save_path=f"predictions/{server_name}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
        
        if not all_alerts:
            print(f"\n✓ All metrics within normal range. No scaling needed.")
        
        return all_alerts
    
    def run_continuous(self, servers=None, interval=300):
        """Run continuous prediction monitoring"""
        import os
        os.makedirs('predictions', exist_ok=True)
        
        if servers is None:
            servers = ["web-server-01", "api-server-01", "db-server-01"]
        
        print(f"\nStarting continuous predictive monitoring...")
        print(f"Checking every {interval} seconds")
        print(f"Monitoring servers: {', '.join(servers)}\n")
        
        import time
        
        try:
            while True:
                for server in servers:
                    alerts = self.monitor_server(server)
                    
                    if alerts:
                        self.alert_history.extend(alerts)
                
                print(f"\n{'='*70}")
                print(f"Next check in {interval} seconds...")
                print(f"Total alerts generated: {len(self.alert_history)}")
                print(f"{'='*70}\n")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nPredictive monitoring stopped.")
            print(f"Total alerts in session: {len(self.alert_history)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("AIOps Infrastructure Monitoring - Phase 2B")
    print("LSTM-based Predictive Scaling Engine")
    print("="*70)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python lstm_predictor.py train <server_name> <metric>")
        print("  python lstm_predictor.py predict <server_name> <metric>")
        print("  python lstm_predictor.py monitor")
        print()
        print("Examples:")
        print("  python lstm_predictor.py train web-server-01 cpu_usage")
        print("  python lstm_predictor.py predict web-server-01 cpu_usage")
        print("  python lstm_predictor.py monitor")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) < 4:
            print("Usage: python lstm_predictor.py train <server_name> <metric>")
            sys.exit(1)
        
        server = sys.argv[2]
        metric = sys.argv[3]
        
        scaler = PredictiveScaler()
        scaler.train_predictor(server, metric)
    
    elif command == "predict":
        if len(sys.argv) < 4:
            print("Usage: python lstm_predictor.py predict <server_name> <metric>")
            sys.exit(1)
        
        server = sys.argv[2]
        metric = sys.argv[3]
        
        scaler = PredictiveScaler()
        result = scaler.predict_and_recommend(server, metric)
        
        if result:
            print(f"\n{'='*70}")
            print("PREDICTION RESULTS")
            print(f"{'='*70}")
            print(f"Server: {result['server']}")
            print(f"Metric: {result['metric']}")
            print(f"Current Value: {result['current_value']:.1f}%")
            print(f"Predicted Peak: {result['max_predicted']:.1f}%")
            print(f"Time to Peak: {result['time_to_peak']:.1f} minutes")
            print()
            print(result['recommendation']['message'])
            
            # Generate visualization
            import os
            os.makedirs('predictions', exist_ok=True)
            scaler.visualize_prediction(
                result, 
                save_path=f"predictions/{server}_{metric}_prediction.png"
            )
    
    elif command == "monitor":
        predictor = ContinuousPredictor()
        predictor.run_continuous()
    
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: train, predict, monitor")