"""Run the entire M0 test suite (T1-T15). Plain-script runner; no pytest dependency."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = [
    'test_t1_t2_accumulation.py',
    'test_t3_batchnorm.py',
    'test_t4_t5_acloss.py',
    'test_t6_t7_t8_c4_verifier.py',
    'test_t9_t10_c2_budget.py',
    'test_t11_t12_operator.py',
    'test_t13_t14_t15_firewall_provenance.py',
    'test_m01_provenance_hashes.py',
    'test_m1_paired_configs.py',
    'test_m12_dev_evaluators.py',
    'test_m13_execution_preflight.py',
    'test_m14_final_hardening.py',
    'test_m14a_execution_harness.py',
    'test_m14b_execution_integrity.py',
]
ALL_OK = True
for t in TESTS:
    r = subprocess.run([sys.executable, os.path.join(HERE, t)], capture_output=True, text=True)
    print(r.stdout, end='')
    if r.returncode != 0:
        print(r.stderr, end='')
        ALL_OK = False
print('=' * 60)
print('M0 SUITE:', 'ALL PASS' if ALL_OK else 'FAILURES PRESENT')
sys.exit(0 if ALL_OK else 1)