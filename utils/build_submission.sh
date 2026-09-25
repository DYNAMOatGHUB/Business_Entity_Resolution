#!/bin/bash
set -e

echo "Creating submission staging directory..."
mkdir -p submission_staging/output
mkdir -p submission_staging/code/business_entity_resolution/src

echo "Copying final outputs..."
cp output/matching_results.tsv submission_staging/output/ || echo "Warning: matching_results.tsv not found"
cp output/candidate_pairs.tsv submission_staging/output/ || echo "Warning: candidate_pairs.tsv not found"

echo "Copying codebase..."
cp -r src/* submission_staging/code/business_entity_resolution/src/

echo "Copying environment and instructions..."
cp requirements.txt submission_staging/code/business_entity_resolution/
cp README.md submission_staging/code/business_entity_resolution/

echo "Copying methodology document..."
cp docs/Documentation_template.md submission_staging/

echo "Zipping the contents..."
cd submission_staging
zip -r ../Team_Submission.zip ./*
cd ..

echo "Cleaning up staging directory..."
rm -rf submission_staging

echo "Done! Team_Submission.zip created successfully."
