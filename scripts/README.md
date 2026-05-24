# Scripts

| Script | Action |
|--------|--------|
| `piper_arm.sh` | Piper arm only — no voice/agent/vision (`hackstorm-piper-arm`) |
| `install_pyorbbecsdk.sh` | Install pyorbbecsdk2 on the Linux robot host for DaBai depth |

```bash
chmod +x scripts/*.sh

# Piper arm (Linux robot host, can0 already UP)
./scripts/piper_arm.sh --no-can-auto-init smoke

# Orbbec depth on robot host
./scripts/install_pyorbbecsdk.sh
```
