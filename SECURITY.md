# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do NOT open a public issue for security vulnerabilities.**

### Contact

- Email: Rbkhan00009@gmail.com
- GitHub: [@rbkhan007](https://github.com/rbkhan007)

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Release**: Depends on severity

## Security Best Practices

### For Users

1. **Keep dependencies updated**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use virtual environments**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Don't run as root/administrator**

4. **Verify model integrity**:
   - Only use GGUF files from trusted sources
   - Check file hashes when available

### For Developers

1. **Never commit secrets**:
   - API keys
   - Passwords
   - Tokens
   - Private keys

2. **Use environment variables** for sensitive data

3. **Validate all inputs** before processing

4. **Follow principle of least privilege**

5. **Run tests before committing**

## Known Security Considerations

### Model Files (.gguf)

- Model files are binary and cannot be scanned for malware
- Only download from trusted sources (HuggingFace official repos)
- Verify file sizes match expected values

### Code Execution

- The CLI can execute code via `run-code` and `exec` commands
- Code runs in the current user's context
- Use with caution and only in trusted environments

### Network Access

- Ollama/OpenAI backends require network access
- Default endpoints are localhost only
- No data is sent to external services unless configured

## Updates

This security policy is updated as needed. Check this file for the latest version.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
