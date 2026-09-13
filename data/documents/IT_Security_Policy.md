# Northwind Dynamics — IT Security Policy

**Policy owner:** Information Security
**Effective:** 1 March 2025

## 1. Account security and passwords

- Passwords must be **at least 14 characters** and include upper case, lower case, a digit and a symbol.
- Passwords are rotated every **180 days**; the previous **10 passwords** cannot be reused.
- **Multi-factor authentication (MFA)** is mandatory for email, VPN, the HR portal and all administrative consoles.
- Never share credentials. IT will **never** ask for your password.

## 2. Acceptable use

- Company devices and accounts are for business use; limited personal use is tolerated if it does not affect work or security.
- Prohibited: installing unlicensed software, disabling endpoint protection, connecting unauthorised USB storage, or using company systems for illegal or harassing activity.

## 3. Data classification

| Class | Examples | Handling |
|---|---|---|
| Public | Marketing pages, press releases | No restriction |
| Internal | Org charts, project plans | Employees only |
| Confidential | Customer data, contracts, source code | Need-to-know; encrypt in transit and at rest |
| Restricted | Credentials, financial records, personal data (PII) | Encryption + access logging + approval to share |

## 4. Devices and encryption

- All laptops use **full-disk encryption** (BitLocker or FileVault).
- Screens must lock automatically after **10 minutes** of inactivity.
- Lost or stolen devices must be reported to the Service Desk **within 1 hour** so the device can be wiped remotely.

## 5. Email and phishing

- Do not click links or open attachments from unknown senders.
- Report suspected phishing with the **"Report Phish"** button in Outlook.
- External emails are tagged **[EXTERNAL]** in the subject line.

## 6. Remote access

- Remote access to internal systems requires the **corporate VPN** with MFA.
- Public Wi-Fi may be used only through the VPN.

## 7. Incident reporting

- Report any suspected security incident (malware, data exposure, account compromise) to **security@northwind.example** or the Service Desk **immediately**, and at most **within 1 hour** of discovery.
- Do not attempt to investigate or remediate on your own; preserve evidence and await Information Security.

## 8. Consequences

Violations may lead to loss of system access and disciplinary action up to and including termination, in line with the Code of Conduct.
