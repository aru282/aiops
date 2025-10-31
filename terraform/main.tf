# =============================================================================
# AIOps Infrastructure Monitoring - Terraform Configuration
# Deploys complete AIOps stack on AWS
# =============================================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# VARIABLES
# =============================================================================

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "aiops-monitoring"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  sensitive   = true
}

variable "pagerduty_key" {
  description = "PagerDuty integration key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "influxdb_token" {
  description = "InfluxDB access token"
  type        = string
  sensitive   = true
}

# =============================================================================
# PROVIDER
# =============================================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# VPC AND NETWORKING
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

data "aws_availability_zones" "available" {
  state = "available"
}

# =============================================================================
# SECURITY GROUPS
# =============================================================================

resource "aws_security_group" "lambda" {
  name        = "${var.project_name}-lambda-sg"
  description = "Security group for Lambda functions"
  vpc_id      = aws_vpc.main.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-lambda-sg"
  }
}

resource "aws_security_group" "ec2" {
  name        = "${var.project_name}-ec2-sg"
  description = "Security group for EC2 instances"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }
  
  ingress {
    from_port   = 8086
    to_port     = 8086
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "InfluxDB access"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-ec2-sg"
  }
}

# =============================================================================
# IAM ROLES AND POLICIES
# =============================================================================

# Lambda Execution Role
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_cloudwatch" {
  name = "${var.project_name}-lambda-cloudwatch-policy"
  role = aws_iam_role.lambda_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeNetworkInterfaces",
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# EC2 Instance Role
resource "aws_iam_role" "ec2_instance" {
  name = "${var.project_name}-ec2-instance-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-instance-profile"
  role = aws_iam_role.ec2_instance.name
}

resource "aws_iam_role_policy" "ec2_cloudwatch" {
  name = "${var.project_name}-ec2-cloudwatch-policy"
  role = aws_iam_role.ec2_instance.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })
}

# =============================================================================
# S3 BUCKETS
# =============================================================================

# ML Models Storage
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-ml-models-${var.environment}"
  
  tags = {
    Name = "${var.project_name}-ml-models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lambda Deployment Packages
resource "aws_s3_bucket" "lambda_code" {
  bucket = "${var.project_name}-lambda-code-${var.environment}"
  
  tags = {
    Name = "${var.project_name}-lambda-code"
  }
}

# =============================================================================
# RDS FOR METRIC DATA (Optional - Alternative to InfluxDB)
# =============================================================================

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.public[*].id
  
  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_db_instance" "metrics" {
  identifier           = "${var.project_name}-metrics-db"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  allocated_storage    = 20
  storage_type         = "gp3"
  
  db_name  = "aiops_metrics"
  username = "aiops_admin"
  password = random_password.db_password.result
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.ec2.id]
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Name = "${var.project_name}-metrics-db"
  }
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

# =============================================================================
# EC2 INSTANCE FOR INFLUXDB
# =============================================================================

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "influxdb" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  
  user_data = templatefile("${path.module}/scripts/influxdb_setup.sh", {
    influxdb_token = var.influxdb_token
  })
  
  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }
  
  tags = {
    Name = "${var.project_name}-influxdb-server"
  }
}

resource "aws_eip" "influxdb" {
  instance = aws_instance.influxdb.id
  domain   = "vpc"
  
  tags = {
    Name = "${var.project_name}-influxdb-eip"
  }
}

# =============================================================================
# LAMBDA FUNCTIONS
# =============================================================================

# Anomaly Detection Lambda
resource "aws_lambda_function" "anomaly_detector" {
  filename         = "lambda_packages/anomaly_detector.zip"
  function_name    = "${var.project_name}-anomaly-detector"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("lambda_packages/anomaly_detector.zip")
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 1024
  
  environment {
    variables = {
      INFLUXDB_URL     = "http://${aws_eip.influxdb.public_ip}:8086"
      INFLUXDB_TOKEN   = var.influxdb_token
      SLACK_WEBHOOK    = var.slack_webhook_url
      PAGERDUTY_KEY    = var.pagerduty_key
      MODEL_BUCKET     = aws_s3_bucket.models.id
    }
  }
  
  vpc_config {
    subnet_ids         = aws_subnet.public[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# LSTM Predictor Lambda
resource "aws_lambda_function" "lstm_predictor" {
  filename         = "lambda_packages/lstm_predictor.zip"
  function_name    = "${var.project_name}-lstm-predictor"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("lambda_packages/lstm_predictor.zip")
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 2048
  
  environment {
    variables = {
      INFLUXDB_URL     = "http://${aws_eip.influxdb.public_ip}:8086"
      INFLUXDB_TOKEN   = var.influxdb_token
      SLACK_WEBHOOK    = var.slack_webhook_url
      MODEL_BUCKET     = aws_s3_bucket.models.id
    }
  }
  
  vpc_config {
    subnet_ids         = aws_subnet.public[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# =============================================================================
# CLOUDWATCH EVENTS (EventBridge)
# =============================================================================

# Scheduled trigger for anomaly detection (every 5 minutes)
resource "aws_cloudwatch_event_rule" "anomaly_detection_schedule" {
  name                = "${var.project_name}-anomaly-detection-schedule"
  description         = "Trigger anomaly detection every 5 minutes"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "anomaly_detection" {
  rule      = aws_cloudwatch_event_rule.anomaly_detection_schedule.name
  target_id = "AnomalyDetectionLambda"
  arn       = aws_lambda_function.anomaly_detector.arn
}

resource "aws_lambda_permission" "allow_cloudwatch_anomaly" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.anomaly_detector.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.anomaly_detection_schedule.arn
}

# Scheduled trigger for predictions (every 10 minutes)
resource "aws_cloudwatch_event_rule" "prediction_schedule" {
  name                = "${var.project_name}-prediction-schedule"
  description         = "Trigger LSTM predictions every 10 minutes"
  schedule_expression = "rate(10 minutes)"
}

resource "aws_cloudwatch_event_target" "prediction" {
  rule      = aws_cloudwatch_event_rule.prediction_schedule.name
  target_id = "PredictionLambda"
  arn       = aws_lambda_function.lstm_predictor.arn
}

resource "aws_lambda_permission" "allow_cloudwatch_prediction" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lstm_predictor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.prediction_schedule.arn
}

# =============================================================================
# CLOUDWATCH LOGS
# =============================================================================

resource "aws_cloudwatch_log_group" "anomaly_detector" {
  name              = "/aws/lambda/${aws_lambda_function.anomaly_detector.function_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "lstm_predictor" {
  name              = "/aws/lambda/${aws_lambda_function.lstm_predictor.function_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "aiops_alerts" {
  name              = "/aws/aiops/alerts"
  retention_in_days = 30
}

# =============================================================================
# CLOUDWATCH DASHBOARD
# =============================================================================

resource "aws_cloudwatch_dashboard" "aiops" {
  dashboard_name = "${var.project_name}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AIOps/Monitoring", "AnomaliesDetected", { stat = "Sum", period = 300 }],
            [".", "AlertsGenerated", { stat = "Sum", period = 300 }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "Anomaly Detection Summary"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", { stat = "Sum", period = 300 }],
            [".", "Errors", { stat = "Sum", period = 300 }],
            [".", "Duration", { stat = "Average", period = 300 }]
          ]
          period = 300
          region = var.aws_region
          title  = "Lambda Performance"
        }
      }
    ]
  })
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "influxdb_public_ip" {
  description = "Public IP of InfluxDB server"
  value       = aws_eip.influxdb.public_ip
}

output "influxdb_url" {
  description = "InfluxDB URL"
  value       = "http://${aws_eip.influxdb.public_ip}:8086"
}

output "rds_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.metrics.endpoint
}

output "lambda_anomaly_detector_arn" {
  description = "ARN of anomaly detector Lambda"
  value       = aws_lambda_function.anomaly_detector.arn
}

output "lambda_lstm_predictor_arn" {
  description = "ARN of LSTM predictor Lambda"
  value       = aws_lambda_function.lstm_predictor.arn
}

output "s3_models_bucket" {
  description = "S3 bucket for ML models"
  value       = aws_s3_bucket.models.id
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.aiops.dashboard_name}"
}

output "setup_complete" {
  description = "Setup instructions"
  value = <<-EOT
    AIOps Infrastructure deployed successfully!
    
    Next steps:
    1. SSH to InfluxDB: ssh ubuntu@${aws_eip.influxdb.public_ip}
    2. Access InfluxDB UI: http://${aws_eip.influxdb.public_ip}:8086
    3. View CloudWatch Dashboard: ${aws_cloudwatch_dashboard.aiops.dashboard_name}
    4. Deploy Lambda code: ./scripts/deploy_lambda.sh
    
    Configuration saved to: terraform.tfstate
  EOT
}