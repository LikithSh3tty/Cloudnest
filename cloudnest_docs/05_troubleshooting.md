# CloudNest — Troubleshooting

## Sync Stuck / Not Progressing
1. Check internet connectivity.
2. Confirm you're not over your plan's storage limit (sync pauses when full).
3. Restart the desktop app.
4. Check Preferences > Network for an active bandwidth cap that may be throttling sync.

## Files Missing After Sync
- Check the Trash within the CloudNest web app (Account > Trash) — deleted files are recoverable for 30 days.
- Confirm selective sync didn't exclude the folder (files may still be cloud-only, not missing).

## "Storage Full" Error
This appears when you've hit your plan's storage cap. Options: delete/archive files, empty Trash (Trash counts toward quota), or upgrade your plan.

## App Won't Launch (Desktop)
- Windows: check for conflicting antivirus rules blocking the CloudNest service.
- macOS: ensure CloudNest has Full Disk Access under System Settings > Privacy & Security.
- Reinstalling generally resolves corrupted local index issues.

## Slow Sync Speeds
Common causes: bandwidth cap set too low, large number of small files (CloudNest batches files under 1 MB, which can slow throughput), or ISP throttling. Wired connections generally outperform Wi-Fi for large initial syncs.

## Login Loop / Repeated Sign-In Prompts
Usually caused by an expired session token after a password change on another device. Sign out of all devices from Account Settings > Security > Active Sessions, then sign back in.
