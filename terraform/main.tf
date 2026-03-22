provider "aws" {
  region = var.region
}

# ─── Networking ───────────────────────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = [var.az]
  }
}

resource "aws_security_group" "analytics" {
  name_prefix = "mda-"
  vpc_id      = data.aws_vpc.default.id
  description = "Market Data Analytics - SSH + optional JupyterLab"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
    description = "SSH"
  }

  # JupyterLab — only opened when expose_jupyter = true.
  # Default: use SSH tunnel (ssh -L 8888:localhost:8888 ubuntu@<ip>) instead.
  dynamic "ingress" {
    for_each = var.expose_jupyter ? [1] : []
    content {
      from_port   = var.jupyter_port
      to_port     = var.jupyter_port
      protocol    = "tcp"
      cidr_blocks = [var.admin_cidr]
      description = "JupyterLab"
    }
  }

  tags = { Name = "mda-sg" }

  lifecycle {
    create_before_destroy = true
  }
}

# ─── S3 Data Bucket ───────────────────────────────────────────────────────────
# Shared between the collector (write) and analytics (read).
# Collector syncs closed parquet files here every 10 minutes via cron.

resource "aws_s3_bucket" "data" {
  bucket        = var.s3_data_bucket
  force_destroy = false # Protect data — must be emptied manually before destroy

  tags = { Name = var.s3_data_bucket, Purpose = "mdl-parquet-data" }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Disabled" # Versioning off — parquet files are immutable once closed
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_intelligent_tiering_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  name   = "parquet-auto-tier"

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }
  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
}

# ─── IAM ─────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "analytics" {
  name               = "mda-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Name = "mda-instance-role" }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.analytics.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Analytics instance: read-only access to the shared S3 bucket
resource "aws_iam_role_policy" "s3_read" {
  name = "mda-s3-read"
  role = aws_iam_role.analytics.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.data.arn
      },
      {
        Sid    = "ReadObjects"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.data.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "cloudwatch_logs" {
  name = "mda-cloudwatch-logs"
  role = aws_iam_role.analytics.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
      Resource = "arn:aws:logs:${var.region}:*:log-group:/mda/*"
    }]
  })
}

resource "aws_iam_instance_profile" "analytics" {
  name = "mda-instance-profile"
  role = aws_iam_role.analytics.name
}

# ─── Collector IAM update (optional) ─────────────────────────────────────────
# Grants the existing collector instance's role write access to the S3 bucket.
# Set collector_instance_id = "" to skip.

resource "aws_iam_role_policy" "collector_s3_write" {
  count = var.collector_instance_id != "" ? 1 : 0

  name = "mdl-s3-sync-write"
  role = "mdl-instance-role" # The collector's existing IAM role name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = ["s3:ListBucket"]
        Resource = aws_s3_bucket.data.arn
      },
      {
        Sid    = "WriteObjects"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.data.arn}/*"
      }
    ]
  })
}

# ─── Storage ─────────────────────────────────────────────────────────────────

# Local parquet cache volume — receives data synced from S3
resource "aws_ebs_volume" "data" {
  availability_zone = var.az
  size              = var.data_volume_size_gb
  type              = "gp3"
  throughput        = 500  # MiB/s — analytics reads are sequential/large-block; higher throughput helps
  iops              = 4000 # Baseline 3000 + extra for concurrent notebook reads
  encrypted         = true

  tags = { Name = "mda-data" }
}

# ─── Compute ─────────────────────────────────────────────────────────────────

resource "aws_instance" "analytics" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.analytics.id]
  iam_instance_profile   = aws_iam_instance_profile.analytics.name
  key_name               = var.key_pair_name != "" ? var.key_pair_name : null
  availability_zone      = var.az

  user_data = templatefile("${path.module}/user_data.sh", {
    aws_region     = var.region
    s3_bucket      = var.s3_data_bucket
    jupyter_port   = var.jupyter_port
  })

  depends_on = [
    aws_iam_role_policy.s3_read,
    aws_s3_bucket.data,
  ]

  root_block_device {
    volume_size           = 50  # OS + conda env + notebooks
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  tags = { Name = "md-analytics" }
}

resource "aws_volume_attachment" "data" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.data.id
  instance_id  = aws_instance.analytics.id
  force_detach = false
}
