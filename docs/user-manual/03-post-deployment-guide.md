# Dynamical Edge Platform - Post-Deployment Management

## Part 3: Post-Deployment Management

Your system is installed and running. This guide covers maintaining it, keeping it updated, monitoring performance, and solving problems when they arise.

---

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Routine Maintenance](#routine-maintenance)
3. [Performance Monitoring](#performance-monitoring)
4. [Software Updates](#software-updates)
5. [Backup & Recovery](#backup--recovery)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Security Best Practices](#security-best-practices)
8. [Getting Support](#getting-support)

---

## Daily Operations

### Morning Startup Checklist

Before each operating day, run through this quick checklist:

```
┌─────────────────────────────────────────────────────────────┐
│  ☀️ DAILY STARTUP CHECKLIST                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. Power on all cameras and verify they're online        │
│  □ 2. Power on gloves and verify connection                 │
│  □ 3. Power on the Orin device                              │
│  □ 4. Open dashboard and verify "IDLE" status               │
│  □ 5. Check all devices show "Online"                       │
│  □ 6. Verify safety zones are displayed on map              │
│  □ 7. Start the system and watch for "OPERATIONAL"          │
│  □ 8. Verify TFLOPS usage is within normal range            │
│                                                              │
│  Duration: ~5 minutes                                        │
└─────────────────────────────────────────────────────────────┘
```

### End-of-Day Shutdown

```
┌─────────────────────────────────────────────────────────────┐
│  🌙 END-OF-DAY SHUTDOWN CHECKLIST                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. Click "STOP SYSTEM" on dashboard                      │
│  □ 2. Wait for status to show "IDLE"                        │
│  □ 3. Check Observability for any errors to investigate     │
│  □ 4. (Optional) Export flight recorder log for records     │
│  □ 5. Power down robot to safe position                     │
│  □ 6. (Optional) Power down cameras if not needed           │
│                                                              │
│  Note: The Orin device can remain powered on 24/7           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Monitoring During Operation

**What to watch for on the Dashboard:**

| Indicator | Normal | Warning | Action Needed |
|-----------|--------|---------|---------------|
| **TFLOPS Usage** | 0-80 | 80-100 | Above 100 |
| **Memory** | 0-24 GB | 24-28 GB | Above 28 GB |
| **Status** | OPERATIONAL | CONNECTING | ERROR |
| **Device Count** | All online | Some offline | Many offline |

**Quick health check URL:**
```
http://[YOUR-IP]:8000/health
```

Should return: `{"status": "healthy", "version": "0.3.2"}`

---

## Routine Maintenance

### Weekly Maintenance Tasks

Perform these tasks once per week:

```
┌─────────────────────────────────────────────────────────────┐
│  📅 WEEKLY MAINTENANCE (Every Monday Morning)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. Review flight recorder for recurring errors           │
│        Observability → Flight Recorder → Last 7 days        │
│                                                              │
│  □ 2. Check camera lens cleanliness                          │
│        Dusty lenses = poor detection = unsafe operation     │
│                                                              │
│  □ 3. Recalibrate gloves if tracking seems off               │
│        Devices → Glove → Calibrate                           │
│                                                              │
│  □ 4. Check cloud sync status                                │
│        Cloud → Verify "Last Sync" is recent                  │
│                                                              │
│  □ 5. Review safety zones (still accurate?)                  │
│        Safety → Check all zones make sense                   │
│                                                              │
│  □ 6. Check for software updates                             │
│        (See "Software Updates" section below)                │
│                                                              │
│  Duration: ~15 minutes                                       │
└─────────────────────────────────────────────────────────────┘
```

### Monthly Maintenance Tasks

```
┌─────────────────────────────────────────────────────────────┐
│  📅 MONTHLY MAINTENANCE (First Monday of Month)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. Full camera calibration check                          │
│        Devices → Each Camera → Calibrate → Verify           │
│                                                              │
│  □ 2. Review and clean up old datasets                       │
│        Training → Datasets → Delete unused datasets          │
│                                                              │
│  □ 3. Check storage space                                    │
│        Training tab shows total storage used                 │
│                                                              │
│  □ 4. Review skill performance                               │
│        Training → Version Control → Check success rates      │
│                                                              │
│  □ 5. Full system backup                                     │
│        (See "Backup & Recovery" section below)               │
│                                                              │
│  □ 6. Clean physical hardware                                │
│        - Dust the Orin device vents                          │
│        - Clean camera lenses                                 │
│        - Check all cable connections                         │
│                                                              │
│  Duration: ~30 minutes                                       │
└─────────────────────────────────────────────────────────────┘
```

### Recalibration Schedule

| Device | How Often | Signs You Need It Now |
|--------|-----------|----------------------|
| **Cameras (Intrinsic)** | Every 6 months | Distorted edges in video |
| **Cameras (Extrinsic)** | After any camera movement | 3D tracking errors |
| **Gloves** | Weekly, or after issues | Fingers don't match real hand |
| **PTZ Presets** | Monthly | Camera not going to right position |

---

## Performance Monitoring

### Understanding the Metrics

**TFLOPS (Tera Floating Point Operations Per Second)**

This measures how hard the computer is working:

```
0 ────────────────────────────────────────── 137 TFLOPS
  │        │              │              │
  │        │              │              └── Maximum (don't exceed)
  │        │              └── Warning zone (above 100)
  │        └── Normal operation (40-80)
  └── Idle (0-10)
```

**Memory Usage**

```
0 ────────────────────────────────────────── 32 GB
  │        │              │              │
  │        │              │              └── Maximum (system will slow)
  │        │              └── Warning (above 28)
  │        └── Normal operation (4-20)
  └── Idle (2-4)
```

### Setting Up Alerts

Currently, you need to monitor the dashboard manually. Here are signs to watch for:

**Performance Warning Signs:**

| Sign | What It Means | What To Do |
|------|---------------|------------|
| TFLOPS stays above 100 | System overloaded | Reduce active cameras or stop training |
| Memory above 28 GB | Memory pressure | Restart the system |
| Frequent frame drops | Network or processing issue | Check camera connections |
| High inference latency (>50ms) | Slow responses | Reduce TFLOPS load |

### Checking Historical Performance

1. Go to **Observability → Flight Recorder**
2. Look for patterns in warnings:
   - Do errors happen at certain times?
   - Are specific components repeatedly failing?
3. Export the log for deeper analysis if needed

---

## Software Updates

### Checking for Updates

**Weekly: Check for platform updates**

1. Open Terminal on the Orin device
2. Run:
   ```bash
   cd ~/edge-platform
   git fetch origin
   git status
   ```

3. If it shows "Your branch is behind", there's an update available

**In the UI:**

1. Go to **Cloud** page
2. Click **[Check for Updates]**
3. If update available, you'll see the new version number

### Applying Updates

> **⚠️ Always update during off-hours when the robot isn't needed**

**Step 1: Stop the system**
```bash
# In Terminal, press Ctrl+C if the platform is running
```

**Step 2: Backup current state**
```bash
cp platform.db platform.db.backup
cp settings.json settings.json.backup
```

**Step 3: Pull updates**
```bash
cd ~/edge-platform
git pull origin main
```

**Step 4: Install any new dependencies**
```bash
./update.sh
```

**Step 5: Restart the platform**
```bash
python -m src.platform.api.main
```

**Step 6: Verify everything works**
1. Open the dashboard
2. Check all devices are online
3. Run a quick test operation

### Update Schedule Recommendations

| Update Type | Frequency | Timing |
|-------------|-----------|--------|
| Security patches | As released | Within 1 week |
| Minor updates | Monthly | End of month |
| Major updates | As needed | Planned downtime window |

---

## Backup & Recovery

### What Needs to Be Backed Up

| Data | Location | Importance |
|------|----------|------------|
| **Database** | `platform.db` | Critical - all configs |
| **Settings** | `settings.json` | Important |
| **Safety Zones** | In database | Critical - safety! |
| **Calibrations** | `/var/lib/dynamical/calibration/` | Important |
| **Skill Cache** | `/var/lib/dynamical/skill_cache/` | Can re-download |
| **Training Data** | `/var/lib/dynamical/training/` | Important if local |

### Performing a Backup

**Quick Backup (settings only):**

```bash
# Create backup folder with date
mkdir -p ~/backups/$(date +%Y-%m-%d)

# Copy critical files
cp platform.db ~/backups/$(date +%Y-%m-%d)/
cp settings.json ~/backups/$(date +%Y-%m-%d)/
cp -r /var/lib/dynamical/calibration ~/backups/$(date +%Y-%m-%d)/
```

**Full Backup (including training data):**

```bash
# Create full backup
tar -czvf ~/backups/full-backup-$(date +%Y-%m-%d).tar.gz \
    platform.db \
    settings.json \
    /var/lib/dynamical/
```

### Automated Backup Script

Create a file called `backup.sh`:

```bash
#!/bin/bash
# Dynamical Edge Platform Backup Script

BACKUP_DIR=~/backups/$(date +%Y-%m-%d)
mkdir -p $BACKUP_DIR

echo "Starting backup..."

# Backup database
cp platform.db $BACKUP_DIR/

# Backup settings
cp settings.json $BACKUP_DIR/

# Backup calibrations
cp -r /var/lib/dynamical/calibration $BACKUP_DIR/

# Keep only last 30 days of backups
find ~/backups -type d -mtime +30 -exec rm -rf {} \;

echo "Backup complete: $BACKUP_DIR"
```

Make it executable and schedule it:
```bash
chmod +x backup.sh
# Add to crontab for daily backups at 2 AM:
crontab -e
# Add this line:
0 2 * * * /home/nvidia/edge-platform/backup.sh
```

### Restoring from Backup

**Step 1: Stop the platform**
```bash
# Press Ctrl+C in Terminal running the platform
```

**Step 2: Restore files**
```bash
# Replace with your backup date
BACKUP_DATE=2024-12-10

# Restore database
cp ~/backups/$BACKUP_DATE/platform.db .

# Restore settings
cp ~/backups/$BACKUP_DATE/settings.json .

# Restore calibrations
cp -r ~/backups/$BACKUP_DATE/calibration /var/lib/dynamical/
```

**Step 3: Restart platform**
```bash
python -m src.platform.api.main
```

---

## Troubleshooting Guide

### Problem: Dashboard Won't Load

**Symptoms:** Browser shows "Connection refused" or blank page

**Solutions:**

1. **Is the platform running?**
   ```bash
   # Check if process is running
   ps aux | grep main.py
   ```

2. **Restart the platform:**
   ```bash
   # Stop any existing process
   pkill -f "python.*main.py"

   # Start fresh
   cd ~/edge-platform
   python -m src.platform.api.main
   ```

3. **Check the port isn't blocked:**
   ```bash
   # See what's using port 8000
   netstat -tlnp | grep 8000
   ```

### Problem: Cameras Not Detected

**Symptoms:** Device list shows no cameras or cameras show "Offline"

**Solutions:**

1. **Verify camera is powered on and networked**
   - Check camera power LED
   - Can you access camera directly via its web interface?

2. **Check network connectivity:**
   ```bash
   # Ping the camera
   ping 192.168.1.XXX  # Replace with camera IP
   ```

3. **Try manual connection:**
   - Devices → Add Camera Manually
   - Enter IP, port, username, password

4. **Verify ONVIF is enabled on camera:**
   - Access camera's web interface
   - Look for "ONVIF" or "Integration" settings
   - Enable ONVIF if disabled

### Problem: Glove Not Tracking Accurately

**Symptoms:** On-screen hand doesn't match real hand movements

**Solutions:**

1. **Recalibrate the glove:**
   - Devices → Glove → [Calibrate]
   - Follow all 4 steps carefully
   - Hold each pose very still

2. **Check glove fit:**
   - Glove should fit snugly
   - Sensors should align with finger joints

3. **Check for interference:**
   - Metal objects can interfere with sensors
   - Move away from computers/monitors during calibration

### Problem: Robot Stops Unexpectedly

**Symptoms:** Robot stops moving for no apparent reason

**Solutions:**

1. **Check safety zones:**
   - Safety → Is the robot in a KEEP_OUT zone?
   - Adjust zones if they're too restrictive

2. **Check for human detection:**
   - Observability → Flight Recorder
   - Look for "Human detected" events
   - Verify no false positives

3. **Check safety sensitivity:**
   - Safety → Lower sensitivity slider slightly
   - (But don't compromise safety!)

4. **Check system load:**
   - Dashboard → Is TFLOPS above 100?
   - High load can cause safety timeouts

### Problem: Skills Not Working

**Symptoms:** Task routing finds no skills, or skills fail to execute

**Solutions:**

1. **Check cloud connection:**
   - Cloud → Is status "Connected"?
   - Click [Sync Now] to refresh skill library

2. **Check skill status:**
   - Skills → Are skills showing as "active"?
   - Deprecated skills won't be selected

3. **Check system is running:**
   - Dashboard → Status should be "OPERATIONAL"

4. **Try simpler task description:**
   - "Pick up cube" instead of "Pick up the red cube on the left"

### Problem: High TFLOPS Usage

**Symptoms:** TFLOPS stays above 100, system feels slow

**Solutions:**

1. **Reduce active cameras:**
   - Fewer cameras = less processing
   - Disable cameras not currently needed

2. **Stop training jobs:**
   - Training → Pause any running jobs

3. **Restart the system:**
   - Sometimes processes don't clean up properly
   - Stop and Start the system

4. **Check for runaway processes:**
   ```bash
   top
   # Look for processes using >80% CPU
   ```

### Problem: Database Errors

**Symptoms:** Errors mentioning "sqlite" or "database locked"

**Solutions:**

1. **Restart the platform:**
   ```bash
   pkill -f "python.*main.py"
   python -m src.platform.api.main
   ```

2. **If that doesn't work, reset database:**
   ```bash
   # Backup first!
   cp platform.db platform.db.broken

   # Reinitialize
   rm platform.db
   python -m src.platform.api.database init
   ```

   > ⚠️ This will lose your settings and safety zones!

---

## Security Best Practices

### API Key Management

The platform uses API keys for authentication.

**To change the API key:**

1. Edit the `.env` file:
   ```bash
   nano .env
   ```

2. Change the API_KEY value:
   ```
   API_KEY=your-new-secure-key-here
   ```

3. Restart the platform

**Good API key practices:**
- Use at least 32 random characters
- Never share your API key
- Change it if you suspect compromise

### Network Security

```
┌─────────────────────────────────────────────────────────────┐
│  🔒 NETWORK SECURITY RECOMMENDATIONS                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ Keep the Orin on a private/secure network                │
│  ✓ Use strong Wi-Fi passwords (WPA3 if available)           │
│  ✓ Don't expose port 8000 to the internet                   │
│  ✓ Change default passwords on all cameras                   │
│  ✓ Keep firmware updated on all network devices             │
│                                                              │
│  ✗ DON'T use port forwarding to access remotely             │
│  ✗ DON'T use default "admin/admin" passwords                │
│  ✗ DON'T connect to untrusted networks                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Physical Security

- Keep the Orin device in a secure location
- Secure cables to prevent disconnection
- Label safety-critical equipment clearly
- Restrict physical access to authorized personnel

---

## Getting Support

### Self-Service Resources

| Resource | URL | For |
|----------|-----|-----|
| **Documentation** | docs.dynamical.ai | Guides & tutorials |
| **Knowledge Base** | support.dynamical.ai/kb | Common issues |
| **Community Forum** | community.dynamical.ai | Peer support |
| **Release Notes** | github.com/dynamical-ai/edge-platform/releases | What's new |

### Collecting Information for Support

Before contacting support, gather this information:

```bash
# 1. Platform version
cat ~/edge-platform/VERSION

# 2. System information
uname -a

# 3. Recent logs (last 100 lines)
tail -100 ~/edge-platform/logs/platform.log > support-logs.txt

# 4. Export flight recorder
# In web UI: Observability → Flight Recorder → Export

# 5. Screenshot of error (if applicable)
```

### Contacting Support

**Email Support:**
- support@dynamical.ai
- Include all collected information
- Describe steps to reproduce the issue

**Emergency Support (Safety Critical):**
- emergency@dynamical.ai
- Phone: +1-XXX-XXX-XXXX
- Available 24/7 for safety-related issues

### Support Ticket Template

```
Subject: [Issue Type] - Brief description

Platform Version: 0.3.2
Device: Jetson AGX Orin 32GB

DESCRIPTION:
What happened?

EXPECTED BEHAVIOR:
What should have happened?

STEPS TO REPRODUCE:
1. Go to...
2. Click on...
3. Error appears...

ATTACHED:
- [ ] Platform logs
- [ ] Flight recorder export
- [ ] Screenshots

URGENCY:
- [ ] Critical (system down)
- [ ] High (major feature broken)
- [ ] Medium (workaround exists)
- [ ] Low (minor inconvenience)
```

---

## Quick Reference: Emergency Procedures

### Emergency Stop

If the robot is behaving dangerously:

1. **Physical E-Stop** (if available): Press the red emergency stop button
2. **Software Stop**: Dashboard → "STOP SYSTEM" button
3. **Power Off**: Hold power button on Orin for 10 seconds
4. **Last Resort**: Disconnect power to robot

### System Recovery

If the platform crashes:

1. Wait 30 seconds
2. Restart: `python -m src.platform.api.main`
3. Verify all devices reconnect
4. Check flight recorder for cause

### Data Recovery

If database is corrupted:

1. Stop platform
2. Try: `cp platform.db.backup platform.db`
3. Restart and verify
4. If no backup, reinitialize database (loses settings)

---

## Maintenance Log Template

Keep a log of maintenance activities:

```
┌─────────────────────────────────────────────────────────────┐
│  MAINTENANCE LOG                                             │
├──────────┬──────────┬───────────────────────────────────────┤
│  Date    │ Type     │ Notes                                 │
├──────────┼──────────┼───────────────────────────────────────┤
│ 12/10/24 │ Weekly   │ Cleaned camera lenses, all devices OK │
│ 12/03/24 │ Weekly   │ Recalibrated right glove              │
│ 12/01/24 │ Monthly  │ Full backup, cleared old datasets     │
│ 11/26/24 │ Update   │ Updated to v0.3.2, no issues          │
└──────────┴──────────┴───────────────────────────────────────┘
```

---

## Appendix: Complete Checklist Reference

### Daily Checklist
```
□ Verify all devices online
□ Check system status is healthy
□ Monitor for unusual TFLOPS usage
□ Review any new warnings in flight recorder
```

### Weekly Checklist
```
□ Review week's flight recorder logs
□ Clean camera lenses
□ Check/recalibrate gloves if needed
□ Verify cloud sync is working
□ Review safety zones
□ Check for software updates
```

### Monthly Checklist
```
□ Full camera calibration check
□ Review and clean up datasets
□ Check storage usage
□ Review skill performance metrics
□ Perform full system backup
□ Clean hardware (dust vents)
□ Check all cable connections
□ Update maintenance log
```

---

*Dynamical Edge Platform v0.3.2 - Post-Deployment Management Guide*
*For support, visit: https://dynamical.ai/support*
