#!/bin/bash

TARGET=${1:-"cms"}
CV_SPLITS_PATH=${2:-"/Users/nja/Desktop/Sinergia/data/tinder_expts/5x5stratified_v2_check/cv_splits.json"}
OUTPUT_DIR=${3:-"/Users/nja/Desktop/Sinergia/data/tinder_expts/5x5stratified_v2_check"}
PROBEMAP_PATH=${4:-"/Users/nja/Library/CloudStorage/Box-Box/Molecular_SysBio/projects/Sinergia_CRC/data/gdc_coad/gene_expression/gencode.v22.annotation.gene.probeMap"}
COLOTYPE_PATH=${5:-"/Users/nja/Desktop/Sinergia/data/colotype_gex/colotype_genes.csv"}
for sampling_method in unaugmented gamma_poisson poisson local_crossover global_crossover smote replacement
    do
        for class_size in max 500 5000
            do
        
                python /Users/nja/public_git/signature_sampling/scripts/data_gen.py \
                --cv_splits_path $CV_SPLITS_PATH \
                --ref_df_path $OUTPUT_DIR/"tcga_colotype_gex_counts.csv" \
                --ref_labels_path $OUTPUT_DIR/"tcga_colotype_gex_labels.csv"\
                --probemap_path $PROBEMAP_PATH \
                --colotype_path $COLOTYPE_PATH \
                --sampling_method $sampling_method \
                --class_size $class_size \
                --target $TARGET \
                --validation_size 0.1 \
                --seed 9 \
                --save_dir $OUTPUT_DIR/$sampling_method/$class_size
            done 
    done




