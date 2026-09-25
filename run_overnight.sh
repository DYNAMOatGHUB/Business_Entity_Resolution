#!/bin/bash
set -e

echo "============================================================"
echo " Starting Full Production Run: TRAIN SPLIT"
echo "============================================================"
PYTHONPATH=. python3 src/run.py --split train --stage all

echo "============================================================"
echo " Starting Full Production Run: TEST SPLIT"
echo "============================================================"
PYTHONPATH=. python3 src/run.py --split test --stage all

echo "All Dhyanesh stages completed! Handoff to Prabhu."
