"""
AIOps Infrastructure Monitoring - Phase 1
Metric Collection, Storage, and Visualization Foundation

Requirements:
pip install influxdb-client pandas numpy matplotlib seaborn flask flask-cors

Run InfluxDB via Docker:
docker run -d -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123456 \
  -e DOCKER_INFLUXDB_INIT_ORG=aiops \
  -e DOCKER_INFLUXDB_INIT_BUCKET=metrics \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token \
  influxdb:2.7
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import random
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    INFLUXDB_URL = "http://localhost:8086"
    INFLUXDB_TOKEN = "my-super-secret-token"
    INFLUXDB_ORG = "aiops"
    INFLUXDB_BUCKET = "metrics"
    
    # Simulated infrastructure
    SERVERS = ["web-server-01", "web-server-02", "api-server-01", 
               "db-server-01", "cache-server-01"]
    
    COLLECTION_INTERVAL = 10  # seconds

# ============================================================================
# METRIC SIMULATOR - Creates Realistic Infrastructure Patterns
# ============================================================================
class MetricSimulator:
    """Generates realistic infrastructure metrics with patterns and anomalies"""
    
    def __init__(self):
        self.time_step = 0
        self.anomaly_probability = 0.02  # 2% chance of anomaly
        
    def generate_cpu_usage(self, server_name, timestamp):
        """
        Simulates CPU usage with:
        - Daily patterns (higher during business hours)
        - Weekly patterns (lower on weekends)
        - Random noise
        - Occasional spikes
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base load varies by server type
        if "web" in server_name:
            base = 30
        elif "api" in server_name:
            base = 45
        elif "db" in server_name:
            base = 60
        else:
            base = 25
        
        # Daily pattern (peak during business hours 9-17)
        daily_pattern = 20 * np.sin((hour - 6) * np.pi / 12)
        
        # Weekly pattern (lower on weekends)
        weekly_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Random noise
        noise = np.random.normal(0, 5)
        
        # Occasional spikes (simulates traffic bursts)
        spike = 0
        if random.random() < self.anomaly_probability:
            spike = random.uniform(20, 40)
        
        cpu = base + daily_pattern * weekly_factor + noise + spike
        return max(0, min(100, cpu))  # Clamp between 0-100
    
    def generate_memory_usage(self, server_name, timestamp):
        """Memory usage - grows slowly over time, resets periodically"""
        hour = timestamp.hour
        
        if "db" in server_name:
            base = 70
        elif "cache" in server_name:
            base = 65
        else:
            base = 50
        
        # Memory leak simulation (gradual increase)
        leak = (self.time_step % 720) * 0.02  # Resets every 2 hours
        
        # Daily restart pattern (memory drops at 3 AM)
        if hour == 3:
            leak *= 0.1
        
        noise = np.random.normal(0, 3)
        
        # Anomaly: sudden memory spike
        spike = 0
        if random.random() < self.anomaly_probability * 0.5:
            spike = random.uniform(15, 25)
        
        memory = base + leak + noise + spike
        return max(0, min(100, memory))
    
    def generate_network_throughput(self, server_name, timestamp):
        """Network throughput in MB/s"""
        hour = timestamp.hour
        
        if "api" in server_name:
            base = 150
        elif "web" in server_name:
            base = 200
        else:
            base = 50
        
        # Traffic pattern (peaks during business hours)
        pattern = 100 * np.sin((hour - 8) * np.pi / 10)
        
        noise = np.random.normal(0, 20)
        
        # DDoS simulation (rare)
        spike = 0
        if random.random() < 0.005:  # 0.5% chance
            spike = random.uniform(300, 500)
        
        throughput = base + pattern + noise + spike
        return max(0, throughput)
    
    def generate_disk_io(self, server_name, timestamp):
        """Disk I/O operations per second"""
        if "db" in server_name:
            base = 5000
        else:
            base = 1000
        
        noise = np.random.normal(0, 500)
        
        # Backup operations (spike at night)
        backup_spike = 0
        if 2 <= timestamp.hour <= 4:
            backup_spike = 3000
        
        io = base + noise + backup_spike
        return max(0, io)

# ============================================================================
# DATA COLLECTOR - Writes Metrics to InfluxDB
# ============================================================================
class MetricCollector:
    """Collects and stores metrics in InfluxDB"""
    
    def __init__(self):
        self.client = InfluxDBClient(
            url=Config.INFLUXDB_URL,
            token=Config.INFLUXDB_TOKEN,
            org=Config.INFLUXDB_ORG
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.simulator = MetricSimulator()
        
    def collect_and_store(self):
        """Collect metrics from all servers and store in InfluxDB"""
        timestamp = datetime.utcnow()
        
        for server in Config.SERVERS:
            # Generate metrics
            cpu = self.simulator.generate_cpu_usage(server, timestamp)
            memory = self.simulator.generate_memory_usage(server, timestamp)
            network = self.simulator.generate_network_throughput(server, timestamp)
            disk_io = self.simulator.generate_disk_io(server, timestamp)
            
            # Create InfluxDB points
            points = [
                Point("cpu_usage")
                    .tag("server", server)
                    .field("value", cpu)
                    .time(timestamp, WritePrecision.NS),
                
                Point("memory_usage")
                    .tag("server", server)
                    .field("value", memory)
                    .time(timestamp, WritePrecision.NS),
                
                Point("network_throughput")
                    .tag("server", server)
                    .field("value", network)
                    .time(timestamp, WritePrecision.NS),
                
                Point("disk_io")
                    .tag("server", server)
                    .field("value", disk_io)
                    .time(timestamp, WritePrecision.NS),
            ]
            
            # Write to InfluxDB
            self.write_api.write(
                bucket=Config.INFLUXDB_BUCKET,
                org=Config.INFLUXDB_ORG,
                record=points
            )
            
            print(f"[{timestamp.strftime('%H:%M:%S')}] {server}: "
                  f"CPU={cpu:.1f}% MEM={memory:.1f}% NET={network:.1f}MB/s IO={disk_io:.0f}")
        
        self.simulator.time_step += 1
    
    def run_continuous(self):
        """Run metric collection continuously"""
        print(f"Starting metric collection every {Config.COLLECTION_INTERVAL}s...")
        print(f"Monitoring {len(Config.SERVERS)} servers\n")
        
        try:
            while True:
                self.collect_and_store()
                time.sleep(Config.COLLECTION_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopping metric collection...")
            self.client.close()

# ============================================================================
# DATA QUERY - Read Metrics from InfluxDB
# ============================================================================
class MetricQuery:
    """Query and analyze metrics from InfluxDB"""
    
    def __init__(self):
        self.client = InfluxDBClient(
            url=Config.INFLUXDB_URL,
            token=Config.INFLUXDB_TOKEN,
            org=Config.INFLUXDB_ORG
        )
        self.query_api = self.client.query_api()
    
    def get_recent_metrics(self, metric_type, server=None, duration="1h"):
        """Get recent metrics for analysis"""
        server_filter = f'|> filter(fn: (r) => r["server"] == "{server}")' if server else ''
        
        query = f'''
        from(bucket: "{Config.INFLUXDB_BUCKET}")
          |> range(start: -{duration})
          |> filter(fn: (r) => r["_measurement"] == "{metric_type}")
          {server_filter}
          |> pivot(rowKey:["_time"], columnKey: ["server"], valueColumn: "_value")
        '''
        
        result = self.query_api.query_data_frame(query, org=Config.INFLUXDB_ORG)
        return result
    
    def get_summary_statistics(self):
        """Get summary stats for all servers"""
        stats = {}
        for metric in ["cpu_usage", "memory_usage", "network_throughput"]:
            df = self.get_recent_metrics(metric, duration="30m")
            if not df.empty:
                stats[metric] = {
                    "mean": df.iloc[:, -5:].mean().mean(),
                    "max": df.iloc[:, -5:].max().max(),
                    "min": df.iloc[:, -5:].min().min()
                }
        return stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("AIOps Infrastructure Monitoring - Phase 1")
    print("Metric Collection & Storage System")
    print("="*70)
    print()
    
    # Test InfluxDB connection
    try:
        client = InfluxDBClient(
            url=Config.INFLUXDB_URL,
            token=Config.INFLUXDB_TOKEN,
            org=Config.INFLUXDB_ORG
        )
        health = client.health()
        print(f"✓ InfluxDB connection: {health.status}")
        print()
        client.close()
    except Exception as e:
        print(f"✗ InfluxDB connection failed: {e}")
        print("\nMake sure InfluxDB is running:")
        print("docker run -d -p 8086:8086 ...")
        exit(1)
    
    # Start collecting metrics
    collector = MetricCollector()
    
    # Generate historical data first (optional)
    print("Generating 1 hour of historical data...")
    for i in range(360):  # 1 hour of data at 10s intervals
        collector.collect_and_store()
        if i % 36 == 0:  # Progress update every 6 minutes
            print(f"Progress: {i//36 * 10} minutes generated")
    
    print("\n✓ Historical data generated")
    print("\nStarting real-time collection...\n")
    
    # Run continuous collection
    collector.run_continuous()