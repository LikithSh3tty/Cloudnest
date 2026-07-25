# CloudNest — Technical Setup

## Installing the Desktop App
1. Download the installer from the CloudNest website for your OS.
2. Run the installer and sign in with your CloudNest credentials.
3. Choose which local folders to sync (default: ~/CloudNest).
4. Initial sync begins automatically; large libraries may take several hours.

## Selective Sync
Under Preferences > Sync, you can deselect specific folders to keep them cloud-only (not stored locally). This is useful for laptops with limited disk space.

## Bandwidth Limits
Desktop app > Preferences > Network lets you cap upload/download speed. Default is unlimited.

## Conflict Resolution
If the same file is edited on two devices while offline, CloudNest creates a conflicted copy on next sync, named `filename (conflicted copy, device, date).ext`. Both versions are preserved; CloudNest does not auto-merge content.

## CLI Tool
CloudNest ships a CLI (`cloudnest-cli`) for scripting uploads, downloads, and backup jobs. Available for Linux and macOS. Common commands:
- `cloudnest-cli sync <folder>`
- `cloudnest-cli backup start <job-name>`
- `cloudnest-cli status`

## API Access
Pro and Team plans include API access for programmatic file operations. API keys are generated under Account Settings > Developer. Rate limit: 1000 requests/hour per key.
