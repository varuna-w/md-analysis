variable "region" {
  description = "AWS region to deploy into. Should match the data-collector region for lowest S3 transfer cost."
  type        = string
  default     = "ca-central-1"
}

variable "az" {
  description = "Availability zone. Must match the collector's AZ if using EBS snapshot-based data sharing."
  type        = string
  default     = "ca-central-1a"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type.
    Memory requirements:
      r6i.xlarge  (4 vCPU /  32 GiB) — minimum; tight for multi-file orderbook analysis
      r6i.2xlarge (8 vCPU /  64 GiB) — recommended; handles 10-15 orderbook files + working DataFrames
      r6i.4xlarge (16 vCPU / 128 GiB) — heavy workloads; full-day orderbook across all symbols
  EOT
  type        = string
  default     = "r6i.2xlarge"
}

variable "ami_id" {
  description = <<-EOT
    Ubuntu 24.04 LTS AMI ID for ca-central-1.
    To find the latest:
      aws ec2 describe-images \
        --owners 099720109477 \
        --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
        --region ca-central-1 \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
        --output text
  EOT
  type        = string
}

variable "key_pair_name" {
  description = "Existing EC2 key pair name for SSH access."
  type        = string
  default     = "md-rnd-v"
}

variable "admin_cidr" {
  description = "CIDR allowed inbound SSH (22). Restrict to your IP for production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "data_volume_size_gb" {
  description = <<-EOT
    Size of the local parquet cache EBS volume in GiB.
    The collector produces ~2 GB/hour of orderbook data.
    500 GiB holds ~10 days of full-exchange data locally.
  EOT
  type        = number
  default     = 500
}

variable "s3_data_bucket" {
  description = <<-EOT
    Name for the S3 bucket used to share parquet files between the collector and analytics instances.
    Must be globally unique. Recommended: mdl-parquet-<account-id>.
    The collector syncs closed parquet files here; the analytics instance reads from it.
  EOT
  type        = string
}

variable "collector_instance_id" {
  description = <<-EOT
    EC2 instance ID of the data-collector (market-data-listener) instance.
    Used to scope S3 write permissions on the collector's IAM role.
    Leave empty to skip the collector IAM policy update.
  EOT
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days."
  type        = number
  default     = 30
}

variable "jupyter_port" {
  description = "Port JupyterLab listens on. Access via SSH tunnel (recommended) or direct if expose_jupyter=true."
  type        = number
  default     = 8888
}

variable "expose_jupyter" {
  description = <<-EOT
    Whether to open the JupyterLab port in the security group.
    false (default) = access via SSH tunnel only: ssh -L 8888:localhost:8888 ubuntu@<ip>
    true            = port 8888 open publicly (protected by token auth only — not recommended for production)
  EOT
  type        = bool
  default     = false
}
