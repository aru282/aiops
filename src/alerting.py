"""
AIOps Infrastructure Monitoring - Phase 3
Slack/PagerDuty Integration & AWS Lambda Deployment

Features:
- Slack webhook integration
- PagerDuty API integration
- AWS Lambda function for serverless alerts
- CloudWatch integration
- Alert routing and escalation

Requirements:
pip install requests boto3
"""

import json
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# ALERT INTEGRATIONS
# ============================================================================

class SlackAlertingService:
    """Send alerts to Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_data: Dict) -> bool:
        """Send formatted alert to Slack"""
        
        # Determine color based on severity
        color_map = {
            'CRITICAL': '#FF0000',  # Red
            'HIGH': '#FF6600',      # Orange
            'MEDIUM': '#FFCC00',    # Yellow
            'LOW': '#00CC00',       # Green
            'INFO': '#0099CC'       # Blue
        }
        
        severity = alert_data.get('severity', 'INFO')
        color = color_map.get(severity, '#808080')
        
        # Build Slack message
        slack_message = {
            "username": "AIOps Monitor",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{severity} Alert: {alert_data.get('title', 'Infrastructure Alert')}",
                    "text": alert_data.get('message', ''),
                    "fields": [
                        {
                            "title": "Server",
                            "value": alert_data.get('server', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Metric",
                            "value": alert_data.get('metric', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert_data.get('current_value', 0):.1f}%",
                            "short": True
                        },
                        {
                            "title": "Predicted Peak",
                            "value": f"{alert_data.get('predicted_value', 0):.1f}%",
                            "short": True
                        }
                    ],
                    "footer": "AIOps Monitoring System",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
        
        # Add action buttons if recommendation exists
        if 'recommendation' in alert_data:
            rec = alert_data['recommendation']
            slack_message['attachments'][0]['actions'] = [
                {
                    "type": "button",
                    "text": "View Dashboard",
                    "url": alert_data.get('dashboard_url', '#')
                },
                {
                    "type": "button",
                    "text": f"Action: {rec.get('action', 'Review')}",
                    "style": "danger" if severity in ['CRITICAL', 'HIGH'] else "primary"
                }
            ]
        
        try:
            response = requests.post(
                self.webhook_url,
                json=slack_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✓ Slack alert sent: {severity}")
                return True
            else:
                print(f"✗ Slack alert failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"✗ Slack alert error: {e}")
            return False
    
    def send_summary(self, alerts: List[Dict]) -> bool:
        """Send daily/hourly summary to Slack"""
        
        if not alerts:
            return True
        
        # Group by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary_text = f"*Alert Summary*\n"
        summary_text += f"Total Alerts: {len(alerts)}\n"
        
        for severity, count in sorted(severity_counts.items()):
            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 
                     'LOW': '🟢', 'INFO': '🔵'}.get(severity, '⚪')
            summary_text += f"{emoji} {severity}: {count}\n"
        
        slack_message = {
            "username": "AIOps Monitor",
            "icon_emoji": ":bar_chart:",
            "text": summary_text
        }
        
        try:
            response = requests.post(self.webhook_url, json=slack_message, timeout=10)
            return response.status_code == 200
        except:
            return False


class PagerDutyAlertingService:
    """Send alerts to PagerDuty"""
    
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.events_url = "https://events.pagerduty.com/v2/enqueue"
    
    def send_alert(self, alert_data: Dict) -> bool:
        """Trigger PagerDuty incident"""
        
        severity_map = {
            'CRITICAL': 'critical',
            'HIGH': 'error',
            'MEDIUM': 'warning',
            'LOW': 'info',
            'INFO': 'info'
        }
        
        severity = alert_data.get('severity', 'INFO')
        
        # Build PagerDuty event
        event = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": f"{alert_data.get('server', 'unknown')}_{alert_data.get('metric', 'unknown')}",
            "payload": {
                "summary": alert_data.get('title', 'Infrastructure Alert'),
                "severity": severity_map.get(severity, 'info'),
                "source": alert_data.get('server', 'Unknown'),
                "custom_details": {
                    "metric": alert_data.get('metric', 'Unknown'),
                    "current_value": alert_data.get('current_value', 0),
                    "predicted_value": alert_data.get('predicted_value', 0),
                    "message": alert_data.get('message', '')
                }
            }
        }
        
        try:
            response = requests.post(
                self.events_url,
                json=event,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 202:
                print(f"✓ PagerDuty alert sent: {severity}")
                return True
            else:
                print(f"✗ PagerDuty alert failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"✗ PagerDuty alert error: {e}")
            return False


class AlertRouter:
    """Route alerts to appropriate channels based on severity"""
    
    def __init__(self, slack_webhook: str = None, pagerduty_key: str = None):
        self.slack = SlackAlertingService(slack_webhook) if slack_webhook else None
        self.pagerduty = PagerDutyAlertingService(pagerduty_key) if pagerduty_key else None
    
    def route_alert(self, alert_data: Dict):
        """Route alert based on severity and configuration"""
        
        severity = alert_data.get('severity', 'INFO')
        
        # Send to Slack for all alerts
        if self.slack:
            self.slack.send_alert(alert_data)
        
        # Send to PagerDuty only for HIGH and CRITICAL
        if self.pagerduty and severity in ['HIGH', 'CRITICAL']:
            self.pagerduty.send_alert(alert_data)
        
        # Log alert
        self._log_alert(alert_data)
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file for audit trail"""
        import os
        os.makedirs('logs', exist_ok=True)
        
        log_file = f"logs/alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')


# ============================================================================
# AWS LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda function handler
    
    This function is triggered by CloudWatch Events or can be invoked directly
    It runs anomaly detection and sends alerts
    
    Environment Variables Required:
    - SLACK_WEBHOOK_URL: Slack webhook URL
    - PAGERDUTY_KEY: PagerDuty integration key
    - INFLUXDB_URL: InfluxDB endpoint
    - INFLUXDB_TOKEN: InfluxDB access token
    """
    
    print(f"AIOps Lambda triggered at {datetime.now()}")
    
    # Load configuration from environment
    slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
    pagerduty_key = os.environ.get('PAGERDUTY_KEY')
    
    # Initialize alert router
    router = AlertRouter(slack_webhook, pagerduty_key)
    
    # Import monitoring components
    # (In Lambda, these would be packaged as dependencies)
    try:
        from anomaly_detector import RealTimeMonitor
        from lstm_predictor import ContinuousPredictor
        
        # Run anomaly detection
        monitor = RealTimeMonitor()
        monitor.detector.load_model()
        
        anomaly_alerts = monitor.monitor_all_servers()
        
        # Run predictive analysis
        predictor = ContinuousPredictor()
        prediction_alerts = []
        
        for server in ["web-server-01", "api-server-01", "db-server-01"]:
            alerts = predictor.monitor_server(server, ['cpu_usage', 'memory_usage'])
            if alerts:
                prediction_alerts.extend(alerts)
        
        # Route all alerts
        total_alerts = 0
        
        for server, alerts in anomaly_alerts.items():
            for alert in alerts:
                router.route_alert({
                    'title': 'Anomaly Detected',
                    'server': server,
                    'severity': alert.get('severity', 'MEDIUM'),
                    'metric': alert.get('primary_metric', 'Unknown'),
                    'current_value': alert.get('metric_value', 0),
                    'message': f"Anomaly detected in {alert.get('primary_metric', 'metrics')}"
                })
                total_alerts += 1
        
        for alert in prediction_alerts:
            rec = alert.get('recommendation', {})
            router.route_alert({
                'title': 'Predictive Scaling Alert',
                'server': alert['server'],
                'severity': rec.get('severity', 'MEDIUM'),
                'metric': alert['metric'],
                'current_value': alert['current_value'],
                'predicted_value': alert['max_predicted'],
                'message': rec.get('message', ''),
                'recommendation': rec
            })
            total_alerts += 1
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed successfully',
                'alerts_generated': total_alerts,
                'timestamp': datetime.now().isoformat()
            })
        }
    
    except Exception as e:
        print(f"Error in Lambda execution: {e}")
        
        # Send error alert
        if slack_webhook:
            error_alert = {
                'title': 'AIOps Lambda Error',
                'severity': 'CRITICAL',
                'message': f'Lambda execution failed: {str(e)}',
                'server': 'lambda',
                'metric': 'execution_error'
            }
            router.route_alert(error_alert)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        }


# ============================================================================
# CLOUDWATCH INTEGRATION
# ============================================================================

class CloudWatchIntegration:
    """Send metrics and logs to AWS CloudWatch"""
    
    def __init__(self):
        import boto3
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        self.log_group = '/aws/aiops/monitoring'
        self.log_stream = f"alerts-{datetime.now().strftime('%Y%m%d')}"
    
    def put_metric(self, metric_name: str, value: float, unit: str = 'None', 
                   dimensions: List[Dict] = None):
        """Send custom metric to CloudWatch"""
        
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.now()
        }
        
        if dimensions:
            metric_data['Dimensions'] = dimensions
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace='AIOps/Monitoring',
                MetricData=[metric_data]
            )
            print(f"✓ CloudWatch metric sent: {metric_name}={value}")
        except Exception as e:
            print(f"✗ CloudWatch metric error: {e}")
    
    def log_alert(self, alert_data: Dict):
        """Log alert to CloudWatch Logs"""
        
        try:
            # Ensure log group and stream exist
            try:
                self.logs.create_log_group(logGroupName=self.log_group)
            except:
                pass  # Group already exists
            
            try:
                self.logs.create_log_stream(
                    logGroupName=self.log_group,
                    logStreamName=self.log_stream
                )
            except:
                pass  # Stream already exists
            
            # Send log
            self.logs.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[
                    {
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'message': json.dumps(alert_data)
                    }
                ]
            )
            print(f"✓ CloudWatch log sent")
        except Exception as e:
            print(f"✗ CloudWatch log error: {e}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_slack_integration():
    """Example: Send alert to Slack"""
    
    # Set your Slack webhook URL
    SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    slack = SlackAlertingService(SLACK_WEBHOOK)
    
    # Example alert
    alert = {
        'title': 'High CPU Usage Predicted',
        'severity': 'HIGH',
        'server': 'web-server-01',
        'metric': 'cpu_usage',
        'current_value': 65.5,
        'predicted_value': 92.3,
        'message': 'CPU usage predicted to reach 92% in 25 minutes',
        'recommendation': {
            'action': 'SCALE_SOON',
            'scale_up_percent': 30
        },
        'dashboard_url': 'https://your-dashboard.com/server/web-server-01'
    }
    
    slack.send_alert(alert)


def example_full_integration():
    """Example: Full integration with routing"""
    
    # Configuration
    SLACK_WEBHOOK = os.environ.get('SLACK_WEBHOOK_URL', '')
    PAGERDUTY_KEY = os.environ.get('PAGERDUTY_KEY', '')
    
    # Initialize router
    router = AlertRouter(SLACK_WEBHOOK, PAGERDUTY_KEY)
    
    # Simulate some alerts
    alerts = [
        {
            'title': 'Memory Leak Detected',
            'severity': 'CRITICAL',
            'server': 'api-server-01',
            'metric': 'memory_usage',
            'current_value': 95.2,
            'message': 'Memory usage at critical levels'
        },
        {
            'title': 'High Network Traffic',
            'severity': 'MEDIUM',
            'server': 'web-server-02',
            'metric': 'network_throughput',
            'current_value': 450.8,
            'message': 'Network traffic above normal'
        }
    ]
    
    for alert in alerts:
        router.route_alert(alert)
    
    print(f"\n✓ Sent {len(alerts)} alerts")


if __name__ == "__main__":
    print("="*70)
    print("AIOps Infrastructure Monitoring - Phase 3")
    print("Alert Integration & Deployment System")
    print("="*70)
    print()
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test-slack":
            example_slack_integration()
        elif command == "test-full":
            example_full_integration()
        else:
            print(f"Unknown command: {command}")
    else:
        print("Usage:")
        print("  python alerting.py test-slack    # Test Slack integration")
        print("  python alerting.py test-full     # Test full integration")
        print()
        print("Environment Variables:")
        print("  SLACK_WEBHOOK_URL - Slack webhook URL")
        print("  PAGERDUTY_KEY - PagerDuty integration key")