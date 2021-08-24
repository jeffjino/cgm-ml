#!/usr/bin/env bash

set -euox pipefail

python eval_main.py --qa_config_module qa_config_height  # takes 14min in CI
python eval_main.py --qa_config_module qa_config_height_clean  # takes 9min in CI
# python eval_main.py --qa_config_module qa_config_height_dropout  # takes 14min in CI
# python eval_main.py --qa_config_module qa_config_height_deep_ensemble  # takes 50min in CI
# python eval_main.py --qa_config_module qa_config_height_mcnn  # takes 19min in CI
# python eval_main.py --qa_config_module qa_config_height_filter

python eval_main.py --qa_config_module qa_config_weight  # takes 13min in CI
# python eval_main.py --qa_config_module qa_config_weight_dropout  # takes 13min in CI


# Combine results
python combine_results.py --model_measurement height
