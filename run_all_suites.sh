#!/bin/bash
# Master runner for the comprehensive A-MemGuard stress test (SUITE 1-3).
# Runs sequentially to avoid API rate-limit contention. Logs per-suite; writes a sentinel at the end.
cd /home/user/AMemGuard
export SILICONFLOW_API_KEY='sk-opuuixpplazmmzlcwkbrbhykxbcuvspflyilovissdhhzgcf'
rm -f /tmp/ALL_SUITES_DONE
log() { echo "===== [$(date +%H:%M:%S)] $1 ====="; }

log "SUITE-1 datasets x stealth x defenses"
python3 bench_run.py --domains opsec,refund,finance,healthcare,cloud,email \
  --stealth overt,stealthy --defenses none,amemguard,auditor \
  --out bench_suite1_results.json > /tmp/suite1.log 2>&1
echo "suite1 exit=$?"

log "SUITE-2 memory systems"
python3 bench_memsys.py --domains opsec,refund --memsys raw,mem0,genagents,memgpt \
  --stealth stealthy --defenses none,amemguard \
  --out bench_suite2_memsys_results.json > /tmp/suite2.log 2>&1
echo "suite2 exit=$?"

log "SUITE-3a multi-agent misinformation"
python3 bench_mas.py --n_agents 5 --out bench_suite3_mas_results.json > /tmp/suite3a.log 2>&1
echo "suite3a exit=$?"

log "SUITE-3b cross-lingual (zh)"
python3 bench_xling.py --out bench_suite3_xling_results.json > /tmp/suite3b.log 2>&1
echo "suite3b exit=$?"

log "SUITE-3c self-reinforcing cycle (seeded poison)"
python3 realenv_cycle.py --rounds 6 --seed_poison --out realenv_cycle_seeded_results.json > /tmp/suite3c.log 2>&1
echo "suite3c exit=$?"

log "ALL DONE"
touch /tmp/ALL_SUITES_DONE
