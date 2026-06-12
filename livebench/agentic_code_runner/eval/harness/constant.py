# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Image related
BUILD_IMAGE_WORKDIR = "images"
INSTANCE_WORKDIR = "instances"
EVALUATION_WORKDIR = "evals"

# Report related
REPORT_FILE = "report.json"
FINAL_REPORT_FILE = "final_report.json"
# Written to an instance's eval dir instead of running docker when the filtered
# fix patch is empty; gen_report routes such instances to empty_patch_ids.
EMPTY_PATCH_MARKER_FILE = "empty-patch.marker"

# Result related
RUN_LOG_FILE = "run.log"
TEST_PATCH_RUN_LOG_FILE = "test-patch-run.log"
FIX_PATCH_RUN_LOG_FILE = "fix-patch-run.log"

# Log related
BUILD_IMAGE_LOG_FILE = "build_image.log"
RUN_INSTANCE_LOG_FILE = "run_instance.log"
RUN_EVALUATION_LOG_FILE = "run_evaluation.log"
GENERATE_REPORT_LOG_FILE = "gen_report.log"
BUILD_DATASET_LOG_FILE = "build_dataset.log"
