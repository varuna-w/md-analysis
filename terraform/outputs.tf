output "instance_id" {
  description = "Analytics EC2 instance ID."
  value       = aws_instance.analytics.id
}

output "public_ip" {
  description = "Public IP of the analytics instance."
  value       = aws_instance.analytics.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance."
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_instance.analytics.public_ip}"
}

output "jupyter_tunnel_command" {
  description = "SSH tunnel command to access JupyterLab locally at http://localhost:8888"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem -L ${var.jupyter_port}:localhost:${var.jupyter_port} ubuntu@${aws_instance.analytics.public_ip}"
}

output "s3_bucket_name" {
  description = "S3 bucket for parquet data sharing between collector and analytics."
  value       = aws_s3_bucket.data.id
}

output "s3_sync_command" {
  description = "Command to run on the collector to sync parquet data to S3 (add to cron on collector)."
  value       = "aws s3 sync /data/ s3://${aws_s3_bucket.data.id}/ --exclude '*.parquet' --include '*/*.parquet' --exclude '*_$(date +%Y%m%d_%H%M)*.parquet'"
}

output "data_volume_id" {
  description = "EBS volume ID for the local parquet cache."
  value       = aws_ebs_volume.data.id
}
