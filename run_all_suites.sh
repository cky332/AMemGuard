#!/bin/bash
# Leaner master runner (serial API ~6s/call). opsec already fully covered separately, so
# SUITE-1 here covers the OTHER domains for breadth. Caps queries for tractability.
cd /home/user/AMemGuard
export SILICONFLOW_API_KEY='sk-opuuixpplazmmzlcwkbrbhykxbcuvspflyilovissdhhzgcf'
rm -f /tmp/ALL_SUITES_DONE
log() { echo "===== [$(date +%H:%M:%S)] $1 ====="; }

# SUITE-1A: full 3-defense comparison on the other 'landing' policy-shaped domains
log "SUITE-1A refund+cloud x stealth x defenses"
MAX_TRIG=4 MAX_BEN=2 python3 bench_run.py --domains refund,cloud \
  --stealth overt,stealthy --defenses none,amemguard,auditor \
  --out bench_suite1_results.json > /tmp/suite1.log 2>&1
echo "suite1a exit=$?"

# SUITE-1B: vulnerability map (No-Defense only) on strong-prior domains
log "SUITE-1B vulnerability map finance+healthcare+email (none)"
MAX_TRIG=4 MAX_BEN=0 python3 bench_run.py --domains finance,healthcare,email \
  --stealth overt,stealthy --defenses none \
  --out bench_suite1b_results.json > /tmp/suite1b.log 2>&1
echo "suite1b exit=$?"

log "SUITE-2 memory systems (opsec x 4 memsys)"
MAX_TRIG=4 python3 bench_memsys.py --domains opsec --memsys raw,mem0,genagents,memgpt \
  --stealth stealthy --defenses none,amemguard \
  --out bench_suite2_memsys_results.json > /tmp/suite2.log 2>&1
echo "suite2 exit=$?"

log "SUITE-3a multi-agent misinformation"
MAX_Q=4 python3 bench_mas.py --n_agents 5 --out bench_suite3_mas_results.json > /tmp/suite3a.log 2>&1
echo "suite3a exit=$?"

log "SUITE-3b cross-lingual (zh)"
python3 bench_xling.py --out bench_suite3_xling_results.json > /tmp/suite3b.log 2>&1
echo "suite3b exit=$?"

log "SUITE-3c self-reinforcing cycle (seeded poison)"
python3 realenv_cycle.py --rounds 6 --seed_poison --out realenv_cycle_seeded_results.json > /tmp/suite3c.log 2>&1
echo "suite3c exit=$?"

log "ALL DONE"; touch /tmp/ALL_SUITES_DONE
