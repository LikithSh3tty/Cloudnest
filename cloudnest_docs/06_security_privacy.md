# CloudNest — Security & Privacy

## Encryption
- Data in transit: TLS 1.2+ for all uploads/downloads.
- Data at rest: AES-256 encryption on CloudNest's servers.
- Optional end-to-end encrypted (E2EE) vaults: only the user holds the decryption key. CloudNest cannot recover files in an E2EE vault if the key is lost — there is no password reset for vault keys.

## Data Residency
Standard accounts store data in US-based data centers. Team plans can opt into EU data residency at signup (cannot be changed after account creation).

## Compliance
CloudNest is SOC 2 Type II certified. It is not currently HIPAA-compliant and should not be used to store protected health information.

## Third-Party Sharing
CloudNest does not sell user data. Shared-link files are accessible to anyone with the link unless password-protection is enabled on the share.

## Security Incident Reporting
Report suspected vulnerabilities to security@cloudnest-example.com. CloudNest does not currently run a public bug bounty program.
