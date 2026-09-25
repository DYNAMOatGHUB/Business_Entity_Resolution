#!/usr/bin/env bash
set -euo pipefail

TEAM_NAME="Hydes"
ZIP_NAME="${TEAM_NAME}_submission.zip"
TMP_DIR=$(mktemp -d)

echo "Building submission package: ${ZIP_NAME}..."

# Create directory structure
mkdir -p "${TMP_DIR}/output"
mkdir -p "${TMP_DIR}/code/business_entity_resolution"

# Copy outputs
cp output/matching_results.tsv "${TMP_DIR}/output/"
cp output/candidate_pairs.tsv "${TMP_DIR}/output/"

# Copy code
cp -r src/ configs/ scripts/ README.md requirements.txt "${TMP_DIR}/code/business_entity_resolution/"

# Copy docs
if [ -f docs/Documentation_template.md ]; then
    cp docs/Documentation_template.md "${TMP_DIR}/"
fi

# Package zip
cd "${TMP_DIR}"
zip -r "${OLDPWD}/${ZIP_NAME}" .
cd "${OLDPWD}"

rm -rf "${TMP_DIR}"
echo "Created ${ZIP_NAME} successfully."
