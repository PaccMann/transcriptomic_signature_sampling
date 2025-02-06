#!/bin/bash

TARGET=${1:-"cms"}
CV_SPLITS_PATH=${2:-"/data/5x5stratified_20percent/cv_splits_strat.json"}
OUTPUT_DIR=${3:-"/data/5x5stratified_20percent"}
PROBEMAP_PATH=${4:-"/data/gdc_coad/gene_expression/gencode.v22.annotation.gene.probeMap"}
COLOTYPE_PATH=${5:-"/data/colotype_genes.csv"}
for sampling_method in gamma_poisson poisson local_crossover global_crossover smote replacement
    do
        for class_size in max 500 5000
            do
        
                python data_gen.py \
                --cv_splits_path $CV_SPLITS_PATH \
                --ref_df_path "/data/tcga_colotype_gex_counts.csv" \
                --ref_labels_path "/data/tcga_colotype_gex_labels.csv"\
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

for sampling_method in unaugmented
    do
        for class_size in None
            do
        
                python data_gen.py \
                --cv_splits_path $CV_SPLITS_PATH \
                --ref_df_path "/data/tcga_colotype_gex_counts.csv" \
                --ref_labels_path "/data/tcga_colotype_gex_labels.csv"\
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



