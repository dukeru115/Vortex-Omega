# NFCS Backup & Disaster Recovery Plan

## Overview

Comprehensive backup and disaster recovery (BDR) strategy for the Neural Field Control System (NFCS) production environments. This document outlines backup procedures, recovery protocols, and disaster response strategies to ensure system continuity and data protection.

**Document Version**: 1.0  
**Last Updated**: 2025-09-15  
**Maintained by**: Team Ω - Neural Field Control Systems Research Group

## 🎯 Backup Strategy

### Backup Categories

#### 1. **Critical System Data**
- **Configuration Files**: All NFCS configuration, environment variables, secrets
- **Constitutional Policies**: Safety frameworks, compliance rules, governance data
- **Cognitive Module States**: Memory systems, learned parameters, adaptation data
- **Mathematical Models**: CGL parameters, Kuramoto coupling configurations
- **User Data**: Session data, preferences, historical interactions

#### 2. **Application Code and Infrastructure**
- **Source Code**: Complete codebase with version control history
- **Docker Images**: Containerized application builds and dependencies
- **Infrastructure as Code**: Terraform, Ansible, deployment configurations
- **Monitoring Configurations**: Grafana dashboards, Prometheus rules, alerting

#### 3. **Operational Data**
- **Logs**: Application logs, security events, audit trails
- **Metrics History**: Performance data, monitoring metrics, analytics
- **Research Data**: Experiment results, simulation outputs, analysis
- **Documentation**: Technical docs, procedures, operational knowledge

### Backup Frequency

| Data Type | Frequency | Retention | Storage |
|-----------|-----------|-----------|---------|
| Critical System Data | Real-time + Hourly | 30 days + 12 months | Primary + Cloud |
| Configuration Files | Daily | 90 days | Multiple locations |
| Application Code | On commit | Permanent | Git + Archives |
| Operational Logs | Continuous | 90 days | Log aggregation |
| Metrics Data | Every 15 minutes | 30 days | Time-series DB |
| Research Data | Daily | 2 years | Research storage |

## 🏗️ Backup Infrastructure

### Storage Tiers

#### Tier 1: Primary Backup (Local)
```bash
# Local backup storage configuration
BACKUP_PRIMARY_PATH="/backup/nfcs"
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_ENCRYPTION=true

# Daily backup script location
/opt/nfcs/scripts/backup-daily.sh
```

#### Tier 2: Secondary Backup (Network)
```bash
# Network attached storage
BACKUP_SECONDARY_NFS="backup-server:/nfcs-backups"
BACKUP_SECONDARY_RSYNC="rsync://backup.internal/nfcs"

# Offsite replication
BACKUP_OFFSITE_S3="s3://nfcs-backups-prod"
BACKUP_OFFSITE_REGION="us-west-2"
```

#### Tier 3: Archive Storage (Cloud)
```bash
# Long-term archival
ARCHIVE_GLACIER_BUCKET="nfcs-archive-glacier"
ARCHIVE_RETENTION_YEARS=7
ARCHIVE_COMPRESSION="gzip"
ARCHIVE_ENCRYPTION="AES256"
```

## 📋 Backup Procedures

### 1. **Automated Daily Backup**

```bash
#!/bin/bash
# /opt/nfcs/scripts/backup-daily.sh
# NFCS Daily Backup Script

set -euo pipefail

# Configuration
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/nfcs/$BACKUP_DATE"
LOG_FILE="/var/log/nfcs/backup-$BACKUP_DATE.log"

echo "Starting NFCS daily backup: $BACKUP_DATE" | tee -a "$LOG_FILE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. System Configuration Backup
echo "Backing up system configuration..." | tee -a "$LOG_FILE"
tar -czf "$BACKUP_DIR/config-$BACKUP_DATE.tar.gz" \
    /opt/nfcs/config/ \
    /opt/nfcs/.env* \
    /etc/nfcs/ \
    --exclude='*.tmp' \
    --exclude='*.log'

# 2. Application Data Backup
echo "Backing up application data..." | tee -a "$LOG_FILE"
tar -czf "$BACKUP_DIR/data-$BACKUP_DATE.tar.gz" \
    /opt/nfcs/data/ \
    /var/lib/nfcs/ \
    --exclude='cache/' \
    --exclude='tmp/'

# 3. Constitutional Framework Backup
echo "Backing up constitutional framework..." | tee -a "$LOG_FILE"
tar -czf "$BACKUP_DIR/constitutional-$BACKUP_DATE.tar.gz" \
    /opt/nfcs/constitutional/ \
    /var/lib/nfcs/policies/ \
    /var/lib/nfcs/governance/

# 4. Cognitive Module States
echo "Backing up cognitive module states..." | tee -a "$LOG_FILE"
tar -czf "$BACKUP_DIR/cognitive-$BACKUP_DATE.tar.gz" \
    /var/lib/nfcs/modules/ \
    /var/lib/nfcs/memory/ \
    /var/lib/nfcs/learned_parameters/

# 5. Database Backup (if applicable)
if systemctl is-active --quiet postgresql; then
    echo "Backing up PostgreSQL database..." | tee -a "$LOG_FILE"
    pg_dump nfcs_production | gzip > "$BACKUP_DIR/database-$BACKUP_DATE.sql.gz"
fi

# 6. Log Files Backup
echo "Backing up recent logs..." | tee -a "$LOG_FILE"
find /var/log/nfcs/ -name "*.log" -mtime -7 | \
    tar -czf "$BACKUP_DIR/logs-$BACKUP_DATE.tar.gz" -T -

# 7. Verify backup integrity
echo "Verifying backup integrity..." | tee -a "$LOG_FILE"
for backup_file in "$BACKUP_DIR"/*.tar.gz; do
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        echo "ERROR: Backup file corrupted: $backup_file" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# 8. Calculate checksums
cd "$BACKUP_DIR"
sha256sum *.tar.gz *.sql.gz > checksums.sha256

# 9. Sync to secondary storage
echo "Syncing to secondary storage..." | tee -a "$LOG_FILE"
rsync -avz "$BACKUP_DIR/" "$BACKUP_SECONDARY_NFS/daily/$BACKUP_DATE/"

# 10. Cleanup old backups
echo "Cleaning up old backups..." | tee -a "$LOG_FILE"
find /backup/nfcs/ -maxdepth 1 -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} \;

echo "Daily backup completed successfully: $BACKUP_DATE" | tee -a "$LOG_FILE"

# Send notification
/opt/nfcs/scripts/notify-backup-complete.sh "$BACKUP_DATE" "SUCCESS"
```

### 2. **Real-time Configuration Backup**

```bash
#!/bin/bash
# /opt/nfcs/scripts/backup-realtime-config.sh
# Real-time configuration backup using inotify

# Monitor configuration changes
inotifywait -m -r -e modify,create,delete /opt/nfcs/config/ |
while read path action file; do
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup changed configuration
    tar -czf "/backup/nfcs/realtime/config-$timestamp.tar.gz" \
        /opt/nfcs/config/
    
    # Log change
    echo "$timestamp: Configuration change detected: $path$file ($action)" \
        >> /var/log/nfcs/config-changes.log
    
    # Sync to secondary storage immediately
    rsync -avz "/backup/nfcs/realtime/config-$timestamp.tar.gz" \
        "$BACKUP_SECONDARY_NFS/realtime/"
done
```

### 3. **Research Data Backup**

```bash
#!/bin/bash
# /opt/nfcs/scripts/backup-research-data.sh
# Research data and experiment results backup

RESEARCH_BACKUP_DIR="/backup/nfcs/research/$(date +%Y%m%d)"
mkdir -p "$RESEARCH_BACKUP_DIR"

# Backup experiment results
tar -czf "$RESEARCH_BACKUP_DIR/experiments-$(date +%Y%m%d).tar.gz" \
    /opt/nfcs/experiments/ \
    /var/lib/nfcs/research_data/

# Backup simulation outputs
tar -czf "$RESEARCH_BACKUP_DIR/simulations-$(date +%Y%m%d).tar.gz" \
    /opt/nfcs/simulation_results/ \
    /var/lib/nfcs/simulation_cache/

# Backup analysis and reports
tar -czf "$RESEARCH_BACKUP_DIR/analysis-$(date +%Y%m%d).tar.gz" \
    /opt/nfcs/reports/ \
    /opt/nfcs/analytics/

# Archive to long-term storage
aws s3 cp "$RESEARCH_BACKUP_DIR/" \
    "s3://nfcs-research-archive/$(date +%Y/%m/%d)/" \
    --recursive \
    --storage-class GLACIER
```

## 🚨 Disaster Recovery Procedures

### Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Component | RTO | RPO | Recovery Method |
|-----------|-----|-----|-----------------|
| Critical System Operations | 15 minutes | 5 minutes | Hot standby |
| Constitutional Framework | 30 minutes | 1 hour | Automated restore |
| Cognitive Modules | 1 hour | 4 hours | State restoration |
| Research Data | 4 hours | 24 hours | Archive restoration |
| Historical Analytics | 24 hours | 7 days | Best effort |

### Disaster Scenarios

#### Scenario 1: Hardware Failure
```bash
# Primary server hardware failure recovery
# Target RTO: 15 minutes

# 1. Activate standby server
ssh standby-server "sudo systemctl start nfcs-orchestrator"

# 2. Update DNS/load balancer
curl -X POST https://lb.internal/api/failover \
    -H "Authorization: Bearer $LB_TOKEN" \
    -d '{"primary": "standby-server.internal"}'

# 3. Restore latest configuration
rsync -avz backup-server:/nfcs-backups/realtime/latest/ \
    /opt/nfcs/config/

# 4. Verify system integrity
/opt/nfcs/scripts/health-check.sh --full

# 5. Resume normal operations
systemctl enable --now nfcs-all-services
```

#### Scenario 2: Data Corruption
```bash
# Data corruption recovery procedure
# Target RTO: 1 hour, RPO: 4 hours

# 1. Stop all NFCS services
systemctl stop nfcs-orchestrator nfcs-api nfcs-workers

# 2. Backup corrupted data for analysis
mv /var/lib/nfcs /var/lib/nfcs.corrupted.$(date +%s)

# 3. Restore from latest clean backup
LATEST_BACKUP=$(ls -t /backup/nfcs/ | head -1)
tar -xzf "/backup/nfcs/$LATEST_BACKUP/data-*.tar.gz" -C /

# 4. Restore cognitive module states
tar -xzf "/backup/nfcs/$LATEST_BACKUP/cognitive-*.tar.gz" -C /

# 5. Restore constitutional framework
tar -xzf "/backup/nfcs/$LATEST_BACKUP/constitutional-*.tar.gz" -C /

# 6. Verify data integrity
/opt/nfcs/scripts/verify-data-integrity.sh

# 7. Restart services with verification
systemctl start nfcs-orchestrator
/opt/nfcs/scripts/post-restore-verification.sh

# 8. Resume full operations
systemctl start nfcs-api nfcs-workers
```

#### Scenario 3: Complete Site Failure
```bash
# Complete site disaster recovery
# Target RTO: 4 hours, RPO: 24 hours

# 1. Activate disaster recovery site
ssh dr-site "sudo /opt/nfcs/scripts/activate-dr-site.sh"

# 2. Restore from offsite backups
aws s3 sync s3://nfcs-backups-prod/latest/ /backup/restore/

# 3. Deploy NFCS infrastructure
cd /opt/nfcs/infrastructure/
terraform apply -var="environment=dr" -auto-approve

# 4. Restore application data
for backup in /backup/restore/*.tar.gz; do
    tar -xzf "$backup" -C /
done

# 5. Update configuration for DR environment
/opt/nfcs/scripts/configure-dr-environment.sh

# 6. Start all services
docker-compose -f docker-compose.dr.yml up -d

# 7. Verify full system operation
/opt/nfcs/scripts/dr-verification-suite.sh

# 8. Update DNS to point to DR site
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch file://dr-dns-update.json
```

### Recovery Verification

#### Post-Recovery Checklist
```bash
#!/bin/bash
# /opt/nfcs/scripts/post-restore-verification.sh

echo "Starting post-recovery verification..."

# 1. System Health Check
echo "1. Checking system health..."
if ! /opt/nfcs/scripts/health-check.sh --critical; then
    echo "CRITICAL: System health check failed"
    exit 1
fi

# 2. Constitutional Framework Verification
echo "2. Verifying constitutional framework..."
if ! curl -s http://localhost:8765/constitutional/health | grep -q "healthy"; then
    echo "ERROR: Constitutional framework not responding"
    exit 1
fi

# 3. Cognitive Modules Check
echo "3. Checking cognitive modules..."
for module in constitutional boundary memory meta_reflection freedom; do
    if ! curl -s "http://localhost:5000/api/modules/$module/status" | grep -q "active"; then
        echo "ERROR: Module $module not active"
        exit 1
    fi
done

# 4. Mathematical Core Verification
echo "4. Testing mathematical core..."
python3 -c "
import sys
sys.path.append('/opt/nfcs/src')
from core.cgl_solver import CGLSolver
from core.enhanced_kuramoto import EnhancedKuramoto
solver = CGLSolver()
kuramoto = EnhancedKuramoto(N=5)
print('Mathematical core functional')
"

# 5. Data Integrity Check
echo "5. Verifying data integrity..."
/opt/nfcs/scripts/verify-data-integrity.sh

# 6. Performance Baseline
echo "6. Running performance baseline..."
python3 /opt/nfcs/src/performance/benchmarks.py --quick

# 7. Security Verification
echo "7. Checking security configuration..."
/opt/nfcs/scripts/security-audit.sh --post-recovery

echo "Post-recovery verification completed successfully"
```

## 📊 Backup Monitoring

### Backup Health Dashboard

```python
#!/usr/bin/env python3
# /opt/nfcs/scripts/backup-monitoring.py

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class BackupMonitor:
    def __init__(self):
        self.backup_dir = Path("/backup/nfcs")
        self.metrics = {}
    
    def check_backup_health(self):
        """Check overall backup system health."""
        return {
            'daily_backups': self._check_daily_backups(),
            'realtime_config': self._check_realtime_config(),
            'offsite_sync': self._check_offsite_sync(),
            'storage_usage': self._check_storage_usage(),
            'backup_integrity': self._check_backup_integrity()
        }
    
    def _check_daily_backups(self):
        """Check if daily backups are current."""
        today = datetime.now().strftime("%Y%m%d")
        daily_backups = list(self.backup_dir.glob(f"{today}_*"))
        
        return {
            'status': 'healthy' if daily_backups else 'missing',
            'count': len(daily_backups),
            'latest': max(daily_backups).name if daily_backups else None
        }
    
    def _check_realtime_config(self):
        """Check realtime configuration backup."""
        realtime_dir = self.backup_dir / "realtime"
        if not realtime_dir.exists():
            return {'status': 'missing', 'last_update': None}
        
        latest_config = max(realtime_dir.glob("config-*.tar.gz"), 
                          default=None, key=lambda p: p.stat().st_mtime)
        
        if latest_config:
            age_hours = (time.time() - latest_config.stat().st_mtime) / 3600
            status = 'healthy' if age_hours < 1 else 'stale'
        else:
            status = 'missing'
            age_hours = None
        
        return {
            'status': status,
            'age_hours': age_hours,
            'last_update': latest_config.name if latest_config else None
        }
    
    def generate_backup_report(self):
        """Generate comprehensive backup status report."""
        health = self.check_backup_health()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self._calculate_overall_status(health),
            'components': health,
            'recommendations': self._generate_recommendations(health),
            'next_actions': self._suggest_next_actions(health)
        }
        
        # Save report
        report_file = Path(f"/var/log/nfcs/backup-report-{datetime.now().strftime('%Y%m%d')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

if __name__ == "__main__":
    monitor = BackupMonitor()
    report = monitor.generate_backup_report()
    print(json.dumps(report, indent=2))
```

### Backup Alerting

```bash
#!/bin/bash
# /opt/nfcs/scripts/backup-alerts.sh

# Check for backup failures and send alerts
ALERT_EMAIL="ops@example.com"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Check if daily backup completed
if [ ! -d "/backup/nfcs/$(date +%Y%m%d)_"* ]; then
    echo "ALERT: Daily backup missing for $(date +%Y%m%d)" | \
        mail -s "NFCS Backup Alert: Daily backup missing" $ALERT_EMAIL
    
    curl -X POST $SLACK_WEBHOOK \
        -H 'Content-type: application/json' \
        --data '{"text":"🚨 NFCS Daily backup missing for '$(date +%Y%m%d)'"}'
fi

# Check backup integrity
for backup in /backup/nfcs/$(date +%Y%m%d)_*/*.tar.gz; do
    if ! tar -tzf "$backup" >/dev/null 2>&1; then
        echo "ALERT: Corrupted backup detected: $backup" | \
            mail -s "NFCS Backup Alert: Corrupted backup" $ALERT_EMAIL
    fi
done

# Check storage space
USAGE=$(df /backup | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$USAGE" -gt 85 ]; then
    echo "ALERT: Backup storage ${USAGE}% full" | \
        mail -s "NFCS Backup Alert: Storage space low" $ALERT_EMAIL
fi
```

## 🔧 Backup Automation

### Cron Configuration

```bash
# /etc/cron.d/nfcs-backups
# NFCS Backup Automation

# Daily full backup at 2 AM
0 2 * * * nfcs /opt/nfcs/scripts/backup-daily.sh

# Hourly configuration backup
0 * * * * nfcs /opt/nfcs/scripts/backup-config-hourly.sh

# Weekly research data backup
0 3 * * 0 nfcs /opt/nfcs/scripts/backup-research-data.sh

# Monthly archive to long-term storage
0 4 1 * * nfcs /opt/nfcs/scripts/backup-monthly-archive.sh

# Backup monitoring every 6 hours
0 */6 * * * nfcs /opt/nfcs/scripts/backup-monitoring.py

# Backup cleanup daily at 5 AM
0 5 * * * nfcs /opt/nfcs/scripts/backup-cleanup.sh
```

### Systemd Services

```ini
# /etc/systemd/system/nfcs-backup-realtime.service
[Unit]
Description=NFCS Real-time Configuration Backup
After=network.target

[Service]
Type=simple
User=nfcs
Group=nfcs
ExecStart=/opt/nfcs/scripts/backup-realtime-config.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📚 Testing and Validation

### Backup Testing Schedule

| Test Type | Frequency | Scope | Success Criteria |
|-----------|-----------|-------|------------------|
| Restore Test | Monthly | Configuration only | Complete restore < 30 min |
| Partial DR Test | Quarterly | Single component | RTO/RPO targets met |
| Full DR Test | Annually | Complete system | Full functionality restored |
| Backup Integrity | Weekly | All backup files | No corruption detected |

### Recovery Testing Procedures

```bash
#!/bin/bash
# /opt/nfcs/scripts/test-disaster-recovery.sh

echo "Starting disaster recovery test..."

# 1. Create test environment
docker-compose -f docker-compose.test.yml up -d test-environment

# 2. Simulate data loss
docker exec test-environment rm -rf /var/lib/nfcs/test-data

# 3. Perform recovery
LATEST_BACKUP=$(ls -t /backup/nfcs/ | head -1)
docker exec test-environment tar -xzf "/backup/$LATEST_BACKUP/data-*.tar.gz" -C /

# 4. Verify recovery
docker exec test-environment /opt/nfcs/scripts/verify-data-integrity.sh

# 5. Performance test
docker exec test-environment python3 /opt/nfcs/src/performance/benchmarks.py --quick

# 6. Cleanup
docker-compose -f docker-compose.test.yml down

echo "Disaster recovery test completed"
```

## 📞 Emergency Contacts

### Escalation Matrix

| Severity | Contact | Phone | Email | Response Time |
|----------|---------|-------|-------|---------------|
| Critical | On-call Engineer | +1-555-0100 | oncall@example.com | 15 minutes |
| High | Technical Lead | +1-555-0101 | tech-lead@example.com | 1 hour |
| Medium | Operations Team | +1-555-0102 | ops@example.com | 4 hours |
| Low | Support Team | +1-555-0103 | support@example.com | 24 hours |

### Communication Channels

- **Primary**: Slack #nfcs-ops
- **Backup**: Microsoft Teams
- **Emergency**: SMS alert system
- **Status Page**: https://status.nfcs.internal

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-09-15 | Initial backup and disaster recovery plan | Team Ω |

---

*This document should be reviewed and updated quarterly or after any significant infrastructure changes. For questions or updates, contact the Operations Team.*

_Last updated: 2025-09-15 by Team Ω_