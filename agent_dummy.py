import time

print("[PY] Dummy agent started")

# Simulação de loop (sem ns3-ai real ainda)
# Aqui vamos só manter o processo vivo para validar execução paralela

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("[PY] Dummy agent stopped")
