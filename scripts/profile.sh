#!/bin/bash
python -m cProfile -o out.prof test_profile.py
python -c "import pstats; p = pstats.Stats('out.prof'); p.sort_stats('time').print_stats(40)"